import datetime
import glob
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import pulp

DRT = 'Daily Rainfall Total (mm)'

START_YEAR = 1980
END_YEAR   = 2016

GLOBAL_CUTOFF = 180.0

MIPGAP = '0.1'

def load_rainfall():
    df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join('changi_dailydata', '*.csv'))])

    # We should only have weather data for Changi.
    counts = df['Station'].value_counts()
    assert counts.shape == (1,)
    assert counts['Changi'] > 0
    assert DRT in df.columns

    def extend_date(row):
        assert row['Year']  - int(row['Year'])  == 0
        assert row['Month'] - int(row['Month']) == 0
        assert row['Day']   - int(row['Day'])   == 0

        return datetime.date(int(row['Year']), int(row['Month']), int(row['Day']))

    df['DATE'] = df.apply(extend_date, axis=1)
    df = df.sort_values(by=['DATE'], axis=0)
    df = df.set_index('DATE')

    return df

def find_max(cutoff, data):
    N = len(data)

    indices = []
    weights = []

    for i in range(2, N):
        indices.append((i, i-1, i-2))
        w = data[i] + data[i-1] + data[i-2]

        if w > cutoff:
            weights.append(w)
        else:
            weights.append(0)

    # print ''
    # print 'Nonzero cumulative sums:'
    # for (i, w) in enumerate(weights):
    #     if w > 0: print 'weights[%d] = %.2f' % (i, w)

    # Nothing to do in the degenerate case.
    if [x for x in weights if x > 0] == []: return None

    prob = pulp.LpProblem('maxcsums', pulp.LpMaximize)

    x_vars = [ pulp.LpVariable('x' + str(i), cat='Binary')
                for (i, _) in enumerate(weights) ]

    prob += reduce(lambda a,b: a+b, [ weights[i]*x_vars[i]
                      for i in range(len(weights))])

    def is_adjacent(x, y):
        assert len(x) == 3
        assert len(y) == 3

        (x0, x1, x2) = x
        return (x0 in y or x1 in y or x2 in y)

    for (i, x) in enumerate(indices):
        for (j, y) in enumerate(indices):
            if x < y:
                if is_adjacent(x, y):
                    prob += x_vars[i] + x_vars[j] <= 1

    pulp.GLPK(options=['--mipgap', MIPGAP]).solve(prob)

    soln = []
    data_ix = {}

    for v in prob.variables():
        this_var = int(v.name[1:])

        if v.varValue == 1: soln.append(this_var)

        data_ix[this_var] = indices[this_var]

    return (data_ix, indices, weights, soln, pulp.value(prob.objective))

def make_c3sums(df):
    assert DRT in df.columns
    assert df.index.name == 'DATE'

    dates = list(df.index)
    data  = df[DRT].tolist()

    N = len(data)

    c3sums = []

    for i in range(2, N):
        c3sums.append((dates[i], data[i] + data[i-1] + data[i-2]))

    c3sums = pd.DataFrame(c3sums)
    c3sums.columns = ['DATE', 'c3sum']
    return c3sums

def go_one_year(cutoff, r):
    daily_rainfall = r[DRT].tolist()

    assert len(daily_rainfall) in [365, 366]

    result = find_max(cutoff, daily_rainfall)

    if result is None: return None

    (data_ix, indices, weights, soln, obj) = result

    obj_check = 0
    for (i, s) in enumerate(soln): obj_check += weights[s]
    assert abs(obj - obj_check) < 1e-8

    day3s = []
    hits  = []

    for (i, s) in enumerate(soln):
        assert data_ix[s][0] > data_ix[s][1]
        assert data_ix[s][1] > data_ix[s][2]

        day3s.append(data_ix[s][0])

        hits += [ (r.index[data_ix[s][0]], r.iloc[data_ix[s][0]][DRT] + r.iloc[data_ix[s][1]][DRT] + r.iloc[data_ix[s][2]][DRT]) ]

    hits = np.array(hits)
    hits = pd.DataFrame(data=hits, index=hits[:,0])
    hits.columns = ['DATE', 'c3sum']

    year_counts = r.iloc[day3s].copy()

    return (len(year_counts.index.value_counts()), hits)

def plot_all_c3sums_not_disjoint():
    df = load_rainfall()

    c3sums = make_c3sums(df)

    f = 'c3sums.png'

    plt.scatter(c3sums['DATE'].values, c3sums['c3sum'].values)
    plt.xlabel('year')
    plt.ylabel('cumulative 3-sum')
    plt.title('All 3-sums (not disjoint).')
    plt.savefig(f)
    plt.close()

    print '\n\n==> ' + f

def compute_cutoffs_table():
    yearly = load_rainfall().groupby(lambda x: x.year)

    cutoff_dicts = {}

    for cutoff in [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300]:
        cutoffs_by_year = {}
        for (y, r) in yearly:
            result = go_one_year(cutoff, r)

            if result is None:
                cutoffs_by_year[y] = 0
            else:
                cutoffs_by_year[y] = result[0]

        cutoff_dicts[cutoff] = cutoffs_by_year

    def values_by_year(x):
        return [x[k] for k in sorted(x.keys())]

    columns = []

    columns.append([''] + [str(y) for y in range(START_YEAR, END_YEAR+1)])

    for c in sorted(cutoff_dicts.keys()):
        columns.append([c] + values_by_year(cutoff_dicts[c]))

    data = np.array(columns).transpose()

    return pd.DataFrame(
                data=data[1:,1:],
                index=data[1:,0],
                columns=data[0,1:])

def compute_table_daily_burn():
    yearly = load_rainfall().groupby(lambda x: x.year)

    df = []

    for (y, r) in yearly:
        x = go_one_year(GLOBAL_CUTOFF, r)
        if x is not None: df.append(x[1])

    df = pd.concat(df)
    df['burn'] = df.apply(lambda row: max(row['c3sum'] - GLOBAL_CUTOFF, 0), axis=1)

    return df

def plot_yearly_burn(daily_burn):
    yearly_burn = daily_burn.copy()
    yearly_burn = yearly_burn.set_index('DATE')
    yearly_burn = yearly_burn.groupby(lambda x: x.year).sum()

    f = 'yearly_burn.png'

    plt.scatter(yearly_burn.index.values, yearly_burn['burn'].values)
    plt.xlabel('year')
    plt.ylabel('burn')
    plt.savefig('yearly_burn.png')
    plt.close()
    print '\n\n==> ' + f

    f = 'yearly_burn_histogram.png'
    yearly_burn.hist(bins=50)
    plt.title('Histogram of yearly burn')
    plt.xlabel('yearly total burn')
    plt.savefig(f)
    plt.close()
    print '\n\n==> ' + f

    f = 'yearly_burn_cumulative.png'
    yearly_burn.hist(bins=50, cumulative=True)
    plt.title('Cumulative histo of yearly burn')
    plt.xlabel('yearly burn')
    plt.savefig(f)
    plt.close()
    print '\n\n==> ' + f

    print 'Expected loss (all years):    ', yearly_burn.mean()
    print 'Expected loss (last 10 years):', yearly_burn.iloc[-10:].mean()
    print 'Expected loss (last 20 years):', yearly_burn.iloc[-20:].mean()
    print 'p99:',                           yearly_burn.quantile(0.99)

if __name__ == '__main__':
    plot_all_c3sums_not_disjoint()

    ctable     = compute_cutoffs_table()
    daily_burn = compute_table_daily_burn()

    plot_yearly_burn(daily_burn)

    print ''
    print ''
    print 'compute_cutoffs_table:'
    print ctable
    print ''
    print ''

    print ''
    print ''
    print 'compute_table_daily_burn:'
    print daily_burn.set_index('DATE')
    print ''
    print ''
