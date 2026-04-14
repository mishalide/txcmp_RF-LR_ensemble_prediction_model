import pandas as pd
import requests
import numpy as np

# config
TBA_KEY = '(PUT IT HERE)'
TBA_URL = "https://www.thebluealliance.com/api/v3"
STATBOTICS_V3_URL = "https://api.statbotics.io/v3"

# change your events
EVENTS = {
    'aldine': '2026txhou',
    'space_city': '2026txcle'
}

def get_tba_oprs(event_key):
    r = requests.get(f"{TBA_URL}/event/{event_key}/oprs", headers={"X-TBA-Auth-Key": TBA_KEY})
    data = r.json()
    # oprs dict: {'frc5414': 20.5, ...}
    tba_oprs = data.get('oprs', {})
    return {int(k.replace('frc', '')): v for k, v in tba_oprs.items()}

def get_statbotics_epas(event_key):
    r = requests.get(f"{STATBOTICS_V3_URL}/team_events?event={event_key}")
    data = r.json()
    epas = {}
    for row in data:
        team = row['team']
        epa_val = row.get('epa', {}).get('norm', 1400) # fallback
        if epa_val is None: epa_val = 1400
        epas[team] = epa_val
    return epas

def parse_pit_scouting(file_path):
    df = pd.read_csv(file_path)
    
    parsed = {}
    for idx, row in df.iterrows():
        try:
            team = int(row['Team #'])
        except (ValueError, TypeError):
            continue
            
        def is_true(v):
            return str(v).strip().upper() == 'TRUE'

        purpose = str(row.get('Primary Purpose of Robot', '')).lower()
        is_fuel = any(x in purpose for x in ['score', 'shoot', 'fuel', 'crammer', 'dumper'])
        
        try:
            d_hrs = float(row.get('Driver Experience (Hours)', 0))
            if np.isnan(d_hrs): d_hrs = 0.0
        except:
            d_hrs = 0.0
            
        try:
            bq = float(row.get('Build Quality', 5))
            if np.isnan(bq): bq = 5.0
        except:
            bq = 5.0
            
        parsed[team] = {
            'climb_l3': 1 if is_true(row.get('L3 Climb Endgame')) else 0,
            'climb_l2': 1 if is_true(row.get('L2 Climb Endgame')) else 0,
            'climb_l1': 1 if is_true(row.get('L1 Climb Endgame')) else 0,
            'auto_align': 1 if is_true(row.get('Auto Alignment')) else 0,
            'depot': 1 if is_true(row.get('Intake from Depot')) else 0,
            'outpost': 1 if is_true(row.get('Intake from Outpost')) else 0,
            'under_trench': 1 if is_true(row.get('Under Trench')) else 0,
            'over_bump': 1 if is_true(row.get('Over Bump')) else 0,
            'fuel_scorer': 1 if is_fuel else 0,
            'driver_hours': d_hrs,
            'build_quality': bq
        }
    return parsed

def build_event_data(event_name, event_key):
    oprs = get_tba_oprs(event_key)
    epas = get_statbotics_epas(event_key)
    
    pit_data = parse_pit_scouting(f"raw/{event_name}_pit_scouting.csv")
    
    def get_pit(team):
        if team in pit_data: return pit_data[team]
        return {
            'climb_l3': 0, 'climb_l2': 0, 'climb_l1': 0, 'auto_align': 0,
            'depot': 0, 'outpost': 0, 'under_trench': 0, 'over_bump': 0,
            'fuel_scorer': 0, 'driver_hours': 0, 'build_quality': 5.0
        }

    schedule = pd.read_csv(f"raw/{event_name}_schedule.csv")
    
    schedule = schedule[schedule['Match'].astype(str).str.startswith('Quals')]
    
    all_rows = []
    
    for _, row in schedule.iterrows():
        try:
            b1 = int(row['Blue 1'])
            b2 = int(row['Blue 2'])
            b3 = int(row['Blue 3'])
            r1 = int(row['Red 1'])
            r2 = int(row['Red 2'])
            r3 = int(row['Red 3'])
            blue_score = float(row['Blue Score'])
            red_score = float(row['Red Score'])
        except (ValueError, TypeError, KeyError):
            continue
            
        b_teams = [b1, b2, b3]
        r_teams = [r1, r2, r3]
        
        red_win = 1 if red_score > blue_score else 0
        
        row_out = {'red_win': red_win}
        
        for color, teams in [('red', r_teams), ('blue', b_teams)]:
            p_data = [get_pit(t) for t in teams]
            
            row_out[f'{color}_opr_sum'] = sum([oprs.get(t, 0) for t in teams])
            row_out[f'{color}_epa_sum'] = sum([epas.get(t, 1400) for t in teams])
            
            row_out[f'{color}_climb_l3_any'] = 1 if any([p['climb_l3'] for p in p_data]) else 0
            row_out[f'{color}_climb_l2_count'] = sum([p['climb_l2'] for p in p_data])
            row_out[f'{color}_climb_l1_count'] = sum([p['climb_l1'] for p in p_data])
            row_out[f'{color}_auto_align_sum'] = sum([p['auto_align'] for p in p_data])
            row_out[f'{color}_depot_sum'] = sum([p['depot'] for p in p_data])
            row_out[f'{color}_outpost_sum'] = sum([p['outpost'] for p in p_data])
            row_out[f'{color}_under_trench_sum'] = sum([p['under_trench'] for p in p_data])
            row_out[f'{color}_over_bump_sum'] = sum([p['over_bump'] for p in p_data])
            row_out[f'{color}_fuel_scorer_count'] = sum([p['fuel_scorer'] for p in p_data])
            row_out[f'{color}_driver_hours_sum'] = sum([p['driver_hours'] for p in p_data])
            row_out[f'{color}_driver_hours_max'] = max([p['driver_hours'] for p in p_data])
            row_out[f'{color}_build_quality_mean'] = np.mean([p['build_quality'] for p in p_data])
            
        all_rows.append(row_out)
        
    return pd.DataFrame(all_rows)

if __name__ == "__main__":
    df_aldine = build_event_data('aldine', EVENTS['aldine'])
    df_space_city = build_event_data('space_city', EVENTS['space_city'])
    
    df_all = pd.concat([df_aldine, df_space_city], ignore_index=True)
    df_all.to_csv('rebuilt_training_data.csv', index=False)
    print(f"Built training data with {len(df_all)} matches.")
