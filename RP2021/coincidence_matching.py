import numba
import numpy as np
import pandas as pd


@numba.njit(cache=True, nogil=True)
def fast_coincidence_match(chan_sel, channels, energies, times,
                           time_window=1000  # ns
                           ):
    """
    Coincidence match data on the basis of the
    :param chan_sel: length 2 array of channel numbers. E.g. np.array([4,5])
    :param channels: array of channels
    :param energies: array of energies
    :param times: array of timestamps in [s]
    :param time_window: max time difference [in ns] between two events to qualify as being
    in coincidence.
    :return: 4 dimensional array of [e0, e1, t0, t1]:
        e0: the energies of events where we also observed something in the other detector
        e1: the energies of the event in that other detector
        t0: the timestamp of events where we also observed something in the other detector
        t1: the timestamp of the event in that other detector
    """
    if not (len(chan_sel) == 2 and chan_sel[1] - chan_sel[0] == 1):
        raise ValueError('channel_numbers not correct format')

    # this is where we will store the results
    e0 = []
    e1 = []
    t0 = []
    t1 = []

    for x in range(len(energies)):
        if channels[x] != chan_sel[0]:
            pass
        else:
            # y is the index that is closest to index x in time
            y = find_closest_other_channel(x, 100, times, channels)
            if y == -1:
                raise ValueError('Did not find closest argument in data')
            # if time difference is smaller than allowed by timewindow (ns!) save this match
            elif np.abs(times[y] - times[x]) < time_window/1e9:
                e0.append(energies[x])
                e1.append(energies[y])
                t0.append(times[x])
                t1.append(times[y])

    return np.array(e0), np.array(e1), np.array(t0), np.array(t1)


@numba.njit(cache=True, nogil=True)
def find_closest_other_channel(x, dx, time, channel):
    """
    Return index y of there the time between times[x] and times[y] is smallest and the channel[x] != channel[y]
    :param x: index of element of interest in times
    :param time: array of times
    :param dx: index y may differ from x by this much
    :param channel: array of len
    :return:
    """
    if len(channel) != len(time):
        raise ValueError('Arrays time and channel should be of same length')
    # Only look between low_bound and high_bound (this assumes time is ordered!)
    low_bound = np.max(np.array([x-dx, 0]))
    high_bound = np.min(np.array([x + dx, len(time)]))
    ys = np.arange(low_bound, high_bound)

    # write the results as res_y (the index of interest). Keep track of previous results in dt
    dt = np.inf
    res_y = -1
    target = time[x]

    # Only loop over small range of indices given by ys
    for y in ys:
        if channel[y] == channel[x]:
            # only save results if channel is different
            continue
        else:
            # Compute dt if smaller than previous -> update result unless x == y
            delta = np.abs(time[y] - target)
            if delta < dt and y != x:
                dt = delta
                res_y = y
    return res_y


def easy_coincidence_matching(data,
                              source='Co60',
                              check_time_order=True):
    """
    For a given data set use fast_coincidence_match to match events based on their
    timestamps to events in another channel to do a coincidence study
    :param data: data set (pandas data-frame)
    :param source: name of the source
    :param check_time_order: check if the data we are selecting is actually time ordered.
    Although unlikely this may give undesired implications in find_closest_other_channel.
    :return: data set of matched peaks
    """
    sources = {'Co60': np.array([4, 5]),
               'Ti44': np.array([2, 3]),
               'Cs137': np.array([6, 7])}
    if source in sources:
        source_channels = sources[source]
    else:
        raise NotImplementedError(
            f'Source {source} is not in this easy matching. Try {source.keys()}')

    # Select the data from this source
    source_data = data[(data['channel'] == source_channels[0]) |
                       (data['channel'] == source_channels[1])]

    #
    if check_time_order:
        print(f'Checking time order of your data. This can take a while. Disable with '
              f'check_time_order = False')
        if np.any(np.diff(source_data.time) < 0):
            print(f'Warning! Your data was not time ordered, doing that now. May take a while')
            source_data = source_data.sort_values('time')

    # Let's do the matching!
    e0, e1, t0, t1 = fast_coincidence_match(
        source_channels,
        channels=source_data['channel'].values,
        energies=source_data['integral'].values,
        times=source_data['time'].values)

    # Convert these results to a slightly more readable format of a pandas data-frame
    result = {
        f'e_ch{source_channels[0]}': np.array(e0),
        f'e_ch{source_channels[1]}': np.array(e1),
        f't_ch{source_channels[0]}': np.array(t0),
        f't_ch{source_channels[1]}': np.array(t1),
    }
    return pd.DataFrame(result)


def select_peak(data,
                key,
                energy=1173.2,
                energy_range=50):
    """
    Give data where the value of key is within energy +/- energy_range
    :param data: pd.DataFrame with key as one of the columns
    :param key: key of the column of interest
    :param energy: energy to select by default the first Co60 photo-peak
    :param energy_range: range around the photo_peak to take into account
    :return: data where the data[key] is within required range
    """
    return data[(energy - energy_range < data[key]) &
                (energy + energy_range > data[key])]
