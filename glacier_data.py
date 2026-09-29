"""
Loaders for the Leclercq et al. (2014) glacier length dataset (supplementary tables S3 and S4).

S3_glacierproperties.txt
    One row per glacier (471), tab-delimited, with a single header row.
    Columns: num, name, ID, region, lat, lon, area (km2), L_1950 (km), hmax (m), hmin (m),
    slope, precip (m/a), #data, first, last.
    Missing values appear as 'NaN' (and one precip value of 9999.00, an obvious fill value).

S4_lengthrecords.txt
    Wide layout: 3 columns per glacier (year, dL (m), source), 3 x 471 = 1413 columns.
    Row 1: glacier name and ID for each column triple (third cell empty).
    Row 2: the sub-header 'year', 'dL (m)', 'source' repeated.
    Rows 3..N+2: data points; records shorter than the longest record are padded with -99999.
    Glaciers appear in the same order as in S3.
    'source' is the method of measurement, categories 1-5 as defined in the main paper.

Both loaders return pandas DataFrames keyed on the glacier ID, so metadata and length records can
be matched with ``records.join(props, on='ID')`` or ``records[records.ID == some_id]``.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / 'supmat_zip'
PROPERTIES_FILE = DATA_DIR / 'S3_glacierproperties.txt'
LENGTHRECORDS_FILE = DATA_DIR / 'S4_lengthrecords.txt'

MISSING = -99999

PROPERTY_COLUMNS = {
    'num': 'num',
    'name': 'name',
    'ID': 'ID',
    'region': 'region',
    'lat': 'lat',
    'lon': 'lon',
    'area (km2)': 'area_km2',
    'L_1950 (km)': 'L1950_km',
    'hmax (m)': 'hmax_m',
    'hmin (m)': 'hmin_m',
    'slope': 'slope',
    'precip (m/a)': 'precip_m_per_a',
    '#data': 'n_data',
    'first': 'first',
    'last': 'last',
}


def load_glacier_properties(path=PROPERTIES_FILE) -> pd.DataFrame:
    '''
    Load glacier metadata (table S3)
    :param path: Path to S3_glacierproperties.txt
    :return: DataFrame indexed by glacier ID, one row per glacier, in file order
    '''
    # skipinitialspace: numbers and 'NaN' are right-aligned with leading spaces (e.g. ' NaN')
    props = pd.read_csv(path, sep='\t', skipinitialspace=True, na_values=['NaN', str(MISSING)])
    # Header cells carry trailing spaces ('name ', 'ID ', ...) and names are space padded
    props.columns = props.columns.str.strip()
    props = props.rename(columns=PROPERTY_COLUMNS)
    props['name'] = props['name'].str.strip()
    # Whisky Bay g has precip = 9999.00, which is a fill value, not a physical precipitation rate
    props.loc[props['precip_m_per_a'] >= 9999, 'precip_m_per_a'] = np.nan
    return props.set_index('ID')


def load_length_records(path=LENGTHRECORDS_FILE) -> pd.DataFrame:
    '''
    Load glacier length records (table S4) into long format
    :param path: Path to S4_lengthrecords.txt
    :return: DataFrame with columns ID, name, year, dL_m, source; one row per data point,
             sorted by glacier (file order) and year
    '''
    with open(path) as f:
        header = f.readline().rstrip('\n').split('\t')
    n_glaciers = len(header) // 3
    names = [s.strip() for s in header[0:3 * n_glaciers:3]]
    ids = [int(s) for s in header[1:3 * n_glaciers:3]]

    raw = pd.read_csv(path, sep='\t', header=None, skiprows=2).to_numpy()
    if raw.shape[1] != 3 * n_glaciers:
        raise ValueError(f'Expected {3 * n_glaciers} data columns, found {raw.shape[1]}')
    # (row, glacier, [year, dL, source])
    triples = raw.reshape(raw.shape[0], n_glaciers, 3)

    frames = []
    for g in range(n_glaciers):
        rec = triples[:, g, :]
        rec = rec[(rec != MISSING).all(axis=1)]
        frames.append(pd.DataFrame({
            'ID': ids[g],
            'name': names[g],
            'year': rec[:, 0].astype(int),
            'dL_m': rec[:, 1].astype(float),
            'source': rec[:, 2].astype(int),
        }))
    records = pd.concat(frames, ignore_index=True)
    # Records are chronological except Lewis (1987 listed before 1986); a stable sort fixes that
    order = {gid: i for i, gid in enumerate(ids)}
    records = records.sort_values(['ID', 'year'], key=lambda s: s.map(order) if s.name == 'ID' else s,
                                  kind='stable', ignore_index=True)
    return records


def load_glacier_dataset(properties_path=PROPERTIES_FILE, records_path=LENGTHRECORDS_FILE):
    '''
    Load both tables and check that they describe the same glaciers
    :return: (properties, records) - see load_glacier_properties and load_length_records
    '''
    props = load_glacier_properties(properties_path)
    records = load_length_records(records_path)

    rec_ids = records['ID'].unique()
    if list(rec_ids) != list(props.index):
        raise ValueError('Glacier IDs/order differ between S3 and S4')
    rec_names = records.groupby('ID', sort=False)['name'].first()
    if not (rec_names == props['name']).all():
        raise ValueError('Glacier names differ between S3 and S4')
    counts = records.groupby('ID', sort=False).size()
    if not (counts == props['n_data']).all():
        raise ValueError('Number of data points in S4 does not match #data in S3')
    return props, records


def get_record(records: pd.DataFrame, glacier) -> pd.DataFrame:
    '''
    Select the length record of a single glacier
    :param records: Long-format records from load_length_records
    :param glacier: Glacier ID (int) or name (str)
    :return: DataFrame with year, dL_m, source for that glacier
    '''
    key = 'name' if isinstance(glacier, str) else 'ID'
    rec = records[records[key] == glacier]
    if rec.empty:
        raise KeyError(f'No glacier with {key} = {glacier!r}')
    return rec[['year', 'dL_m', 'source']].reset_index(drop=True)
