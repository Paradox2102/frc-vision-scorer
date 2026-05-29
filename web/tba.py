"""
The Blue Alliance API v3 client.
Fetches event info, match schedules, team lists, and score breakdowns.
"""

import os
import re
import httpx
from typing import Optional, List, Dict, Any

TBA_BASE_URL = "https://www.thebluealliance.com/api/v3"


def get_api_key() -> str:
    """Get TBA API key from environment."""
    key = os.environ.get("TBA_API_KEY", "")
    if not key:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
        key = os.environ.get("TBA_API_KEY", "")
    return key


def parse_event_key(url_or_key: str) -> str:
    """Extract event key from a TBA URL or raw key.

    Examples:
        'https://www.thebluealliance.com/event/2026week0' -> '2026week0'
        '2026week0' -> '2026week0'
        'thebluealliance.com/event/2026casd' -> '2026casd'
    """
    # Try to extract from URL
    match = re.search(r'(?:thebluealliance\.com/event/)([a-zA-Z0-9_]+)', url_or_key)
    if match:
        return match.group(1)
    # Assume it's already a key
    return url_or_key.strip().strip('/')


async def tba_get(endpoint: str) -> Any:
    """Make an authenticated GET request to TBA API."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError("TBA_API_KEY not configured. Add it to .env file.")

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TBA_BASE_URL}{endpoint}",
            headers={"X-TBA-Auth-Key": api_key},
            timeout=15.0
        )
        response.raise_for_status()
        return response.json()


async def get_event(event_key: str) -> Dict:
    """Get event details."""
    return await tba_get(f"/event/{event_key}")


async def get_event_teams(event_key: str) -> List[Dict]:
    """Get teams at an event."""
    return await tba_get(f"/event/{event_key}/teams")


async def get_event_matches(event_key: str) -> List[Dict]:
    """Get all matches for an event."""
    return await tba_get(f"/event/{event_key}/matches")


async def get_match(match_key: str) -> Dict:
    """Get detailed match data including score breakdown."""
    return await tba_get(f"/match/{match_key}")


def format_comp_level(comp_level: str) -> str:
    """Human-readable comp level."""
    return {
        'qm': 'Quals',
        'sf': 'Semis',
        'f': 'Finals',
        'ef': 'Eighths',
        'qf': 'Quarters',
    }.get(comp_level, comp_level.upper())


def format_match_label(match: Dict) -> str:
    """Generate a human-readable match label like 'Quals 1' or 'Finals 1-2'."""
    level = format_comp_level(match.get('comp_level', ''))
    num = match.get('match_number', 0)
    set_num = match.get('set_number', 1)

    if match.get('comp_level') in ('sf', 'f', 'qf', 'ef'):
        return f"{level} {set_num}-{num}" if set_num > 1 else f"{level} {num}"
    return f"{level} {num}"


def extract_hub_scores(score_breakdown: Dict, alliance: str) -> Dict:
    """Extract hub scoring details from TBA score breakdown."""
    if not score_breakdown or alliance not in score_breakdown:
        return {}

    breakdown = score_breakdown[alliance]
    hub = breakdown.get('hubScore', {})

    return {
        'total': hub.get('totalCount', 0),
        'auto': hub.get('autoCount', 0),
        'teleop': hub.get('teleopCount', 0),
        'transition': hub.get('transitionCount', 0),
        'shift1': hub.get('shift1Count', 0),
        'shift2': hub.get('shift2Count', 0),
        'shift3': hub.get('shift3Count', 0),
        'shift4': hub.get('shift4Count', 0),
        'endgame': hub.get('endgameCount', 0),
    }


async def load_event_data(url_or_key: str) -> Dict:
    """Load complete event data: info, teams, and matches.

    Returns a structured dict ready for the frontend.
    """
    event_key = parse_event_key(url_or_key)

    # Fetch event info and matches in parallel
    event_info = await get_event(event_key)
    matches_raw = await get_event_matches(event_key)
    teams_raw = await get_event_teams(event_key)

    # Build team lookup
    teams = {}
    for team in teams_raw:
        num = str(team['team_number'])
        teams[num] = {
            'number': num,
            'name': team.get('nickname', team.get('name', f'Team {num}')),
            'key': team['key'],
        }

    # Build match list
    matches = []
    for m in sorted(matches_raw, key=lambda x: (
        {'qm': 0, 'sf': 1, 'f': 2}.get(x['comp_level'], 3),
        x.get('set_number', 1),
        x['match_number']
    )):
        blue_teams = [k.replace('frc', '') for k in m['alliances']['blue']['team_keys']]
        red_teams = [k.replace('frc', '') for k in m['alliances']['red']['team_keys']]

        blue_hub = extract_hub_scores(m.get('score_breakdown'), 'blue')
        red_hub = extract_hub_scores(m.get('score_breakdown'), 'red')

        # Get YouTube video key if available
        video_key = None
        for v in m.get('videos', []):
            if v.get('type') == 'youtube':
                video_key = v['key']
                break

        matches.append({
            'key': m['key'],
            'comp_level': m['comp_level'],
            'match_number': m['match_number'],
            'set_number': m.get('set_number', 1),
            'label': format_match_label(m),
            'blue_teams': blue_teams,
            'red_teams': red_teams,
            'blue_score': m['alliances']['blue']['score'],
            'red_score': m['alliances']['red']['score'],
            'blue_hub': blue_hub,
            'red_hub': red_hub,
            'video_key': video_key,
        })

    return {
        'event_key': event_key,
        'event_name': event_info.get('name', event_key),
        'year': event_info.get('year', 2026),
        'location': f"{event_info.get('city', '')}, {event_info.get('state_prov', '')}".strip(', '),
        'start_date': event_info.get('start_date'),
        'end_date': event_info.get('end_date'),
        'teams': teams,
        'matches': matches,
    }
