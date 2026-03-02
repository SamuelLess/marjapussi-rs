from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    # All point-based rewards are normalized by this constant.
    points_normalizer: float = 420.0
    # Reward used for pass games (no explicit playing_party contract winner/loser).
    passgame_base_reward: float = 115.0 / 420.0
    # Per-step point-delta scaling for dense intermediate reward.
    step_delta_scale: float = 1.0 / 420.0


def evaluated_team_points(info: dict) -> tuple[float, float]:
    """
    Compute final team points under the project rule-set:
    - Non-playing party keeps all raw team points it made.
    - Playing party gets exactly +game_value on success, else -game_value.
    - Schwarz special-case applies only against the playing party on failure:
      playing party gets -2 * game_value.
    - Pass game (no playing_party): raw team points.
    """
    team_points = info.get("team_points", [0, 0])
    raw0 = float(team_points[0])
    raw1 = float(team_points[1])

    playing_party_raw = info.get("playing_party")
    if playing_party_raw is None:
        return raw0, raw1

    playing_party = int(playing_party_raw)
    opp_party = 1 - playing_party
    won_contract = bool(info.get("won", False))
    game_value = float(info.get("game_value", 0))
    schwarz = bool(info.get("schwarz", False))

    playing_eval = game_value if won_contract else -game_value
    if (not won_contract) and schwarz:
        playing_eval = -2.0 * game_value

    if playing_party == 0:
        return playing_eval, (raw1 if opp_party == 1 else raw0)
    return (raw0 if opp_party == 0 else raw1), playing_eval


def pov_team_points(info: dict, pov_party: int) -> tuple[float, float]:
    team_points = info.get("team_points", [0, 0])
    t0 = float(team_points[0])
    t1 = float(team_points[1])
    return (t0, t1) if pov_party == 0 else (t1, t0)


def pov_team_points_evaluated(info: dict, pov_party: int) -> tuple[float, float]:
    t0, t1 = evaluated_team_points(info)
    return (t0, t1) if pov_party == 0 else (t1, t0)


def point_delta_reward(prev_obs: dict, next_obs: dict, cfg: RewardConfig) -> float:
    my_diff = float(next_obs.get("points_my_team", 0)) - float(prev_obs.get("points_my_team", 0))
    opp_diff = float(next_obs.get("points_opp_team", 0)) - float(prev_obs.get("points_opp_team", 0))
    return (my_diff - opp_diff) * cfg.step_delta_scale


def contract_reward_from_pov(
    info: dict,
    pov_party: int,
    cfg: RewardConfig,
) -> tuple[float, int, int, int | None]:
    """
    Return (terminal_reward_from_pov, tricks_party_0, tricks_party_1, playing_party_abs).
    Reward is aligned to contract outcome (including Schwarz multiplier), not raw trick points.
    """
    tricks_party_0 = sum(1 for t in info.get("tricks", []) if t["winner"] % 2 == 0)
    tricks_party_1 = sum(1 for t in info.get("tricks", []) if t["winner"] % 2 == 1)

    playing_party_raw = info.get("playing_party")
    playing_party_abs = None if playing_party_raw is None else int(playing_party_raw)
    if playing_party_abs is None:
        pov_pts, opp_pts = pov_team_points(info, pov_party)
        if pov_pts > opp_pts:
            return cfg.passgame_base_reward, tricks_party_0, tricks_party_1, None
        if opp_pts > pov_pts:
            return -cfg.passgame_base_reward, tricks_party_0, tricks_party_1, None
        return 0.0, tricks_party_0, tricks_party_1, None

    pov_eval, _opp_eval = pov_team_points_evaluated(info, pov_party)
    total = pov_eval / cfg.points_normalizer
    return total, tricks_party_0, tricks_party_1, playing_party_abs
