from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- FastAPI app -------------------------------------------------------------

openapi_tags = [
    {
        "name": "System",
        "description": "Service health and basic information.",
    },
    {
        "name": "Games",
        "description": "Create games (PvP/PvE), place ships, take turns, and fetch state.",
    },
]

app = FastAPI(
    title="Battleships Backend API",
    description=(
        "Backend for a Battleships web game. Provides game creation, ship placement, "
        "turn-based actions, and state retrieval for PvP and PvE (simple AI)."
    ),
    version="0.1.0",
    openapi_tags=openapi_tags,
)

# Allow the React dev server to call the API (and keep permissive for preview).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Models (Pydantic) -------------------------------------------------------


class Coord(BaseModel):
    """A 0-indexed board coordinate."""

    row: int = Field(..., ge=0, le=9, description="Row index, 0-9.")
    col: int = Field(..., ge=0, le=9, description="Column index, 0-9.")


class ShipPlacement(BaseModel):
    """Represents a single ship placement instruction."""

    start: Coord = Field(..., description="Starting coordinate for the ship (top/left-most).")
    orientation: Literal["H", "V"] = Field(..., description="H for horizontal, V for vertical.")
    length: int = Field(..., ge=2, le=5, description="Ship length. Standard set: 5,4,3,3,2.")


class PlaceShipsRequest(BaseModel):
    """Request payload for placing a player's fleet."""

    player_id: str = Field(..., description="Player id returned by game creation.")
    ships: List[ShipPlacement] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Fleet placements. Must include lengths [5,4,3,3,2] in any order.",
    )


class CreateGameRequest(BaseModel):
    """Create a new game in PvP or PvE mode."""

    mode: Literal["pvp", "pve"] = Field(..., description="Game mode: pvp or pve.")
    board_size: int = Field(10, ge=10, le=10, description="Board size; currently fixed at 10.")
    # For future: player names, etc. Keep minimal for MVP.


class CreateGameResponse(BaseModel):
    """Response from creating a new game."""

    game_id: str = Field(..., description="Unique game id.")
    player1_id: str = Field(..., description="Player 1 id.")
    player2_id: Optional[str] = Field(None, description="Player 2 id (PvP only).")
    mode: Literal["pvp", "pve"] = Field(..., description="Game mode.")


class JoinGameRequest(BaseModel):
    """Join an existing PvP game as player2."""

    game_id: str = Field(..., description="Game id to join.")


class JoinGameResponse(BaseModel):
    """Response when joining a PvP game."""

    game_id: str = Field(..., description="Game id.")
    player2_id: str = Field(..., description="Player 2 id.")


class FireRequest(BaseModel):
    """Request payload to fire at the opponent."""

    player_id: str = Field(..., description="Firing player id.")
    coord: Coord = Field(..., description="Target coordinate.")


class FireResult(BaseModel):
    """Outcome of a fire action."""

    result: Literal["hit", "miss", "sunk", "repeat"] = Field(..., description="Shot outcome.")
    at: Coord = Field(..., description="Target coordinate.")
    sunk_ship_length: Optional[int] = Field(None, description="Length of ship sunk, if any.")
    winner_player_id: Optional[str] = Field(None, description="Winner id, if the game ended.")


class CellView(BaseModel):
    """A single cell as presented to a client."""

    row: int = Field(..., description="Row index.")
    col: int = Field(..., description="Column index.")
    value: Literal["unknown", "miss", "hit", "ship"] = Field(
        ..., description="Cell value (ship only shown on your own board)."
    )


class GameStateResponse(BaseModel):
    """Full game state for a specific requesting player."""

    game_id: str = Field(..., description="Game id.")
    mode: Literal["pvp", "pve"] = Field(..., description="Game mode.")
    you_are: Literal["P1", "P2"] = Field(..., description="Which side the requester is.")
    phase: Literal["placement", "battle", "finished"] = Field(..., description="Game phase.")
    your_turn: bool = Field(..., description="Whether it is your turn to fire.")
    winner: Optional[Literal["P1", "P2"]] = Field(None, description="Winner side if finished.")
    your_board: List[List[str]] = Field(..., description="10x10 view of your board (includes ships).")
    opponent_board: List[List[str]] = Field(
        ..., description="10x10 view of opponent board (unknown/miss/hit only)."
    )
    remaining_ships_you: int = Field(..., description="How many of your ships are still afloat.")
    remaining_ships_opponent: int = Field(..., description="How many opponent ships are still afloat.")
    message: Optional[str] = Field(None, description="Human-friendly status message.")


# --- In-memory game engine ---------------------------------------------------

BOARD_SIZE = 10
FLEET_LENGTHS = [5, 4, 3, 3, 2]


def _new_id(prefix: str, n: int = 8) -> str:
    return prefix + "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(n))


def _in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE


def _coords_for_ship(start: Coord, orientation: str, length: int) -> List[Tuple[int, int]]:
    dr, dc = (0, 1) if orientation == "H" else (1, 0)
    coords: List[Tuple[int, int]] = []
    for i in range(length):
        rr = start.row + dr * i
        cc = start.col + dc * i
        if not _in_bounds(rr, cc):
            raise ValueError("Ship would go out of bounds.")
        coords.append((rr, cc))
    return coords


@dataclass
class Ship:
    length: int
    cells: Set[Tuple[int, int]]
    hits: Set[Tuple[int, int]] = field(default_factory=set)

    def is_sunk(self) -> bool:
        return self.cells.issubset(self.hits)


@dataclass
class PlayerState:
    player_id: str
    # Board stores only ship presence as a set; hits/misses tracked separately.
    ships: List[Ship] = field(default_factory=list)
    ship_cells: Set[Tuple[int, int]] = field(default_factory=set)
    hits_taken: Set[Tuple[int, int]] = field(default_factory=set)  # opponent fired and hit
    misses_taken: Set[Tuple[int, int]] = field(default_factory=set)  # opponent fired and missed
    shots_made: Set[Tuple[int, int]] = field(default_factory=set)  # your fired coords (hit/miss)

    def remaining_ships(self) -> int:
        return sum(1 for s in self.ships if not s.is_sunk())


@dataclass
class Game:
    game_id: str
    mode: Literal["pvp", "pve"]
    player1: PlayerState
    player2: PlayerState
    # placement_done tracks per side
    p1_ready: bool = False
    p2_ready: bool = False
    turn: Literal["P1", "P2"] = "P1"
    winner: Optional[Literal["P1", "P2"]] = None

    def phase(self) -> Literal["placement", "battle", "finished"]:
        if self.winner is not None:
            return "finished"
        if self.p1_ready and self.p2_ready:
            return "battle"
        return "placement"


GAMES: Dict[str, Game] = {}


def _get_game(game_id: str) -> Game:
    game = GAMES.get(game_id)
    if not game:
        raise HTTPException(status_code=404, detail="Game not found.")
    return game


def _side_for_player(game: Game, player_id: str) -> Literal["P1", "P2"]:
    if player_id == game.player1.player_id:
        return "P1"
    if player_id == game.player2.player_id:
        return "P2"
    raise HTTPException(status_code=404, detail="Player not found in this game.")


def _player_state(game: Game, side: Literal["P1", "P2"]) -> PlayerState:
    return game.player1 if side == "P1" else game.player2


def _opponent_side(side: Literal["P1", "P2"]) -> Literal["P1", "P2"]:
    return "P2" if side == "P1" else "P1"


def _validate_fleet_lengths(ships: List[ShipPlacement]) -> None:
    lengths = sorted([s.length for s in ships])
    if lengths != sorted(FLEET_LENGTHS):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fleet lengths. Expected {FLEET_LENGTHS} (in any order).",
        )


def _place_fleet(player: PlayerState, ships: List[ShipPlacement]) -> None:
    _validate_fleet_lengths(ships)

    used: Set[Tuple[int, int]] = set()
    built: List[Ship] = []
    for sp in ships:
        try:
            coords = _coords_for_ship(sp.start, sp.orientation, sp.length)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        # No overlap
        if any(c in used for c in coords):
            raise HTTPException(status_code=400, detail="Ships cannot overlap.")

        # Simple rule: no adjacency constraint for MVP. (Can add later.)
        used.update(coords)
        built.append(Ship(length=sp.length, cells=set(coords)))

    player.ships = built
    player.ship_cells = used
    player.hits_taken.clear()
    player.misses_taken.clear()
    player.shots_made.clear()


def _apply_shot(defender: PlayerState, target: Tuple[int, int]) -> Tuple[str, Optional[int]]:
    """Return (result, sunk_length)."""
    if target in defender.hits_taken or target in defender.misses_taken:
        return "repeat", None

    if target in defender.ship_cells:
        defender.hits_taken.add(target)
        sunk_len: Optional[int] = None
        for ship in defender.ships:
            if target in ship.cells:
                ship.hits.add(target)
                if ship.is_sunk():
                    sunk_len = ship.length
                    return "sunk", sunk_len
                break
        return "hit", None

    defender.misses_taken.add(target)
    return "miss", None


def _ai_pick_shot(attacker: PlayerState, defender: PlayerState) -> Tuple[int, int]:
    """
    Very simple AI:
    - If there is a known hit that hasn't been followed up, try adjacent cells first.
    - Otherwise random among remaining cells.
    """
    # Infer AI knowledge from defender.hits_taken/misses_taken from AI perspective?
    # In our state model, hits_taken/misses_taken are stored on the defender (shots received).
    # AI attacker can look at what it has already shot using attacker.shots_made.
    # We also can look at defender.hits_taken to find cells that were hit by AI only if we track per-attacker,
    # but for MVP we treat attacker.shots_made as "already tried" and use defender.hits_taken to find
    # hit cells (since only one opponent shoots at you). Good enough for PvE.
    hit_cells = list(defender.hits_taken)

    candidates: List[Tuple[int, int]] = []
    for (r, c) in hit_cells:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if _in_bounds(rr, cc) and (rr, cc) not in attacker.shots_made:
                candidates.append((rr, cc))

    if candidates:
        return random.choice(candidates)

    remaining = [(r, c) for r in range(BOARD_SIZE) for c in range(BOARD_SIZE) if (r, c) not in attacker.shots_made]
    return random.choice(remaining)


def _board_view_for_owner(player: PlayerState) -> List[List[str]]:
    grid = [["unknown" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for (r, c) in player.ship_cells:
        grid[r][c] = "ship"
    for (r, c) in player.misses_taken:
        grid[r][c] = "miss"
    for (r, c) in player.hits_taken:
        grid[r][c] = "hit"
    return grid


def _board_view_for_opponent(defender: PlayerState) -> List[List[str]]:
    grid = [["unknown" for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    for (r, c) in defender.misses_taken:
        grid[r][c] = "miss"
    for (r, c) in defender.hits_taken:
        grid[r][c] = "hit"
    return grid


def _state_for_player(game: Game, player_id: str) -> GameStateResponse:
    side = _side_for_player(game, player_id)
    you = _player_state(game, side)
    opp = _player_state(game, _opponent_side(side))

    phase = game.phase()
    msg = None
    if phase == "placement":
        msg = "Place your fleet to begin."
    elif phase == "battle":
        msg = "Your turn!" if game.turn == side else "Opponent's turn."
    else:
        msg = "You won!" if game.winner == side else "You lost."

    return GameStateResponse(
        game_id=game.game_id,
        mode=game.mode,
        you_are=side,
        phase=phase,
        your_turn=(phase == "battle" and game.turn == side),
        winner=game.winner,
        your_board=_board_view_for_owner(you),
        opponent_board=_board_view_for_opponent(opp),
        remaining_ships_you=you.remaining_ships(),
        remaining_ships_opponent=opp.remaining_ships(),
        message=msg,
    )


# --- Routes ------------------------------------------------------------------


@app.get(
    "/",
    tags=["System"],
    summary="Health check",
    description="Simple health check endpoint to verify the backend is running.",
)
def health_check():
    """Return a basic health payload."""
    return {"message": "Healthy"}


@app.get(
    "/rules",
    tags=["System"],
    summary="Game rules / API usage",
    description="Returns a short guide for the Battleships API and default rules.",
)
def rules():
    """Return basic rules and API usage hints."""
    return {
        "boardSize": BOARD_SIZE,
        "fleet": FLEET_LENGTHS,
        "notes": [
            "Coordinates are 0-indexed: row/col in [0..9].",
            "Place exactly 5 ships with lengths [5,4,3,3,2].",
            "In PvE, player2 is an AI and auto-places ships upon game creation.",
        ],
    }


@app.post(
    "/games",
    response_model=CreateGameResponse,
    tags=["Games"],
    summary="Create a new game (PvP or PvE)",
    description="Creates a new game. For PvE, an AI opponent is created and auto-places ships.",
    operation_id="create_game",
)
def create_game(req: CreateGameRequest) -> CreateGameResponse:
    """Create a new game and return ids for players."""
    if req.board_size != BOARD_SIZE:
        raise HTTPException(status_code=400, detail="Only 10x10 boards are supported in this MVP.")

    game_id = _new_id("g_")
    p1_id = _new_id("p1_")
    p2_id = _new_id("p2_")

    game = Game(
        game_id=game_id,
        mode=req.mode,
        player1=PlayerState(player_id=p1_id),
        player2=PlayerState(player_id=p2_id),
    )

    if req.mode == "pve":
        # Auto-place AI fleet randomly; mark p2 ready.
        ships: List[ShipPlacement] = []
        for length in FLEET_LENGTHS:
            placed = False
            while not placed:
                orient = random.choice(["H", "V"])
                start = Coord(
                    row=random.randint(0, BOARD_SIZE - (length if orient == "V" else 1)),
                    col=random.randint(0, BOARD_SIZE - (length if orient == "H" else 1)),
                )
                candidate = ShipPlacement(start=start, orientation=orient, length=length)
                try:
                    # check overlap vs currently planned ships
                    coords = _coords_for_ship(candidate.start, candidate.orientation, candidate.length)
                    used = set()
                    for s in ships:
                        used.update(_coords_for_ship(s.start, s.orientation, s.length))
                    if any(c in used for c in coords):
                        continue
                    ships.append(candidate)
                    placed = True
                except ValueError:
                    continue

        _place_fleet(game.player2, ships)
        game.p2_ready = True

    GAMES[game_id] = game

    return CreateGameResponse(
        game_id=game_id,
        player1_id=p1_id,
        player2_id=(None if req.mode == "pve" else p2_id),
        mode=req.mode,
    )


@app.post(
    "/games/{game_id}/join",
    response_model=JoinGameResponse,
    tags=["Games"],
    summary="Join a PvP game as player2",
    description="Joins a previously created PvP game. The creator already has player2_id; this endpoint is for simple matchmaking flows.",
    operation_id="join_game",
)
def join_game(game_id: str, req: JoinGameRequest) -> JoinGameResponse:
    """Join an existing PvP game (MVP matchmaking)."""
    if req.game_id != game_id:
        raise HTTPException(status_code=400, detail="game_id mismatch between path and body.")
    game = _get_game(game_id)
    if game.mode != "pvp":
        raise HTTPException(status_code=400, detail="Only PvP games can be joined.")
    return JoinGameResponse(game_id=game_id, player2_id=game.player2.player_id)


@app.post(
    "/games/{game_id}/place-ships",
    tags=["Games"],
    summary="Place ships for a player",
    description="Submits a fleet placement for the given player. After both players are ready, battle begins.",
    operation_id="place_ships",
)
def place_ships(game_id: str, req: PlaceShipsRequest):
    """Place ships for a player and advance phase when both are ready."""
    game = _get_game(game_id)
    side = _side_for_player(game, req.player_id)
    player = _player_state(game, side)

    if game.phase() != "placement":
        raise HTTPException(status_code=400, detail="Ships can only be placed during placement phase.")

    _place_fleet(player, req.ships)
    if side == "P1":
        game.p1_ready = True
    else:
        game.p2_ready = True

    # If PvE and P1 just placed, ensure battle starts with P1.
    if game.mode == "pve" and game.p1_ready and game.p2_ready:
        game.turn = "P1"

    return _state_for_player(game, req.player_id).model_dump()


@app.get(
    "/games/{game_id}/state",
    response_model=GameStateResponse,
    tags=["Games"],
    summary="Get game state for a player",
    description="Returns game state tailored to a requesting player. Opponent ships are hidden (unknown) unless hit.",
    operation_id="get_game_state",
)
def get_game_state(game_id: str, player_id: str) -> GameStateResponse:
    """Fetch the current game state (per player view)."""
    game = _get_game(game_id)
    return _state_for_player(game, player_id)


@app.post(
    "/games/{game_id}/fire",
    response_model=FireResult,
    tags=["Games"],
    summary="Fire at the opponent",
    description="Fires at the opponent. In PvE, the AI will automatically respond after the player's shot (if game not finished).",
    operation_id="fire",
)
def fire(game_id: str, req: FireRequest) -> FireResult:
    """Fire a shot for the requesting player and, if PvE, run AI response."""
    game = _get_game(game_id)
    side = _side_for_player(game, req.player_id)

    if game.phase() != "battle":
        raise HTTPException(status_code=400, detail="Cannot fire until both players have placed ships.")
    if game.turn != side:
        raise HTTPException(status_code=400, detail="It is not your turn.")
    if game.winner is not None:
        raise HTTPException(status_code=400, detail="Game is already finished.")

    attacker = _player_state(game, side)
    defender_side = _opponent_side(side)
    defender = _player_state(game, defender_side)

    target = (req.coord.row, req.coord.col)
    if target in attacker.shots_made:
        # We return repeat but do not switch turn.
        return FireResult(result="repeat", at=req.coord, sunk_ship_length=None, winner_player_id=None)

    attacker.shots_made.add(target)
    result, sunk_len = _apply_shot(defender, target)

    # Check win
    if defender.remaining_ships() == 0:
        game.winner = side
        return FireResult(
            result=("sunk" if result == "sunk" else result),
            at=req.coord,
            sunk_ship_length=sunk_len,
            winner_player_id=req.player_id,
        )

    # Switch turn
    game.turn = defender_side

    # PvE auto-response if player1 fired (P1 is human, P2 is AI)
    if game.mode == "pve" and game.turn == "P2" and game.winner is None:
        ai_attacker = game.player2
        ai_defender = game.player1

        ai_target = _ai_pick_shot(ai_attacker, ai_defender)
        ai_attacker.shots_made.add(ai_target)
        _apply_shot(ai_defender, ai_target)

        if ai_defender.remaining_ships() == 0:
            game.winner = "P2"
        else:
            game.turn = "P1"

    return FireResult(
        result=("sunk" if result == "sunk" else result),
        at=req.coord,
        sunk_ship_length=sunk_len,
        winner_player_id=(req.player_id if game.winner == side else None),
    )
