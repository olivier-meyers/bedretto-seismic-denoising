import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import numpy as np

from urllib.parse import urlparse
import pandas as pd

from obspy.clients.fdsn.header import FDSNTimeoutException
from obspy.clients.fdsn.header import FDSNRequestTooLargeException
from obspy.clients.fdsn.header import FDSNNoDataException
import csv

import matplotlib.pyplot as plt

import seisbench.data as sbd
import seisbench.util as sbu

from obspy.clients.fdsn import Client

from pathlib import Path

DATA_DIR = Path(os.environ.get("BEDRETTO_DATA_DIR", "data"))

def mag_to_size(scaler, mag):
  sigma = 1e6 # [Pa] stress drop assumption, could also be higher than 1MPa
  M0 = 10**((mag + 6.03)*3/2) # [Nm] Scalar seismic moment from magnitude (Kanamori)
  SR = (M0 * 7/16 * 1/sigma)**(1/3) # [m]  Source radii from seismic moment and stress drop (Keilis-Borok, 1959)
  return SR * scaler  # scale the physical unit by something sensible for the plot


def create_client(url):
  parsed = urlparse(url)
  base_url = parsed._replace(netloc=f"{parsed.hostname}:{parsed.port}").geturl()
  if parsed.username and parsed.password:
      print(f"Connecting to {base_url} with credentials", file=sys.stderr)
      return Client(base_url=base_url, user=parsed.username, password=parsed.password)
  else:
      print(f"Connecting to {base_url}", file=sys.stderr)
      return Client(base_url=base_url)


def rearrange_picks_to_2d_list(input_list, sorted_list):
  """
  This function distributes the picks dependent on the waveform_id, meaning that P and S picks originating
  from the same trace are arranged in 1 row of a 2D array. Like this, P and S picks for the same trace are
  stored together in one row of the seisbench metadata .csv
  """
  input_list_2D = {}
  sorted_list_2D = {}
  for index, entry in enumerate(input_list):
      if entry not in input_list_2D:
          input_list_2D[entry] = [entry]
          sorted_list_2D[entry] = [sorted_list[index]]
      else:
          input_list_2D[entry].append(entry)
          sorted_list_2D[entry].append(sorted_list[index])

  sorted_list_2D = list(sorted_list_2D.values())
  input_list_2D = list(input_list_2D.values())
  return input_list_2D, sorted_list_2D


def find_exact_string(string_to_find, string_list):
    matching_indices = [i for i, s in enumerate(string_list) if s == string_to_find]
    return matching_indices[0] if matching_indices else []


def main():
    # Load manual picked catalog
    catfile = DATA_DIR / "train_v01/catalog_2022_events.csv"
    # Load M0 dd-catalog
    print(f"Loading catalog {catfile}...", file=sys.stderr)
    cat = pd.read_csv(catfile, dtype=str, na_filter=False)
    print(f"Found {len(cat)} events", file=sys.stderr)

    # Restricted BedrettoLab FDSN server, e.g. "http://<user>:<password>@<host>:<port>"
    client = create_client(os.environ["BEDRETTO_FDSN_URL"])

    window_p_pick = [0.1, 0.05]

    # Seisbench initialization
    base_path = DATA_DIR
    metadata_path = base_path / "metadata.csv"
    waveforms_path = base_path / "waveforms.hdf5"

    pick_count = 0
    P_count = 0
    S_count = 0

    # Iterate over events and picks, write to SeisBench format
    with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
        # , missing_components = "ignore"
        # Define data format
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "Z",
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted"}

        for index, ev_cat in cat.iterrows():
            # if ev_cat.id == 50:
            #     x = 2
            # if index == 152:
            #     x = 2
            print('event: ' + str(index + 1) + '/' + str(len(cat)))
            origin_cat = client.get_events(eventid=ev_cat['ev_id'], includeallorigins=False, includearrivals=True)
            o = origin_cat.events[0].preferred_origin()

            waveform_id_list = []
            pick_hints = []
            picks = []
            arrival_distances = []

            for arrival in o.arrivals:
                pick = None
                for p in origin_cat.events[0].picks:
                    if p.resource_id == arrival.pick_id:
                        pick = p
                        picks.append(pick)
                        waveform_id_list.append(pick.waveform_id.id)
                        pick_hints.append(pick.phase_hint)
                        arrival_distances.append(arrival.distance)
                        break
                if pick is None:
                    print('Arrival without pick (!?)')
                    continue

            waveform_id_list_2D, picks_2D = rearrange_picks_to_2d_list(waveform_id_list, picks)
            waveform_id_list_2D, pick_hints_2D = rearrange_picks_to_2d_list(waveform_id_list, pick_hints)
            waveform_id_list_2D, arrival_distances_2D = rearrange_picks_to_2d_list(waveform_id_list, arrival_distances)

            # no loop over all picks in 2D list
            for index, picks_per_trace in enumerate(picks_2D):

                ind_Pg = find_exact_string('Pg', pick_hints_2D[index]) #changed from 'P' to 'Pg' to match the pick hints in the catalog
                #ind_Pg = find_exact_string('P', pick_hints_2D[index])

                ind_Sg = find_exact_string('Sg', pick_hints_2D[index])


                if type(ind_Pg) is not int and type(ind_Sg) is not int:  # if no picks
                    continue

                if type(ind_Pg) is not int :  # if no P-pick
                    # calculate theoretical time between picks
                    distance = arrival_distances_2D[index][
                                   ind_Sg] * 111.195079734632 * 1000  # [distances in degree, so km/degree * m/km = m]
                    v_p = 5300.0  # [m/s]
                    v_s = 3100.0  # [m/s]
                    delta_p_s = distance * (1 / v_s - 1 / v_p)

                    pick = picks_per_trace[ind_Sg]
                    trace_name = pick.time.isoformat().replace('-', '_').replace('.', '_').replace(':', '_')
                    Pg_theoretical = pick.time - delta_p_s
                    trace_start_time = Pg_theoretical - window_p_pick[0]
                    trace_end_time = Pg_theoretical + window_p_pick[1]

                    signal_window_start_time = pick.time
                    signal_window_end_time = pick.time + delta_p_s

                    noise_window_start_time = Pg_theoretical - delta_p_s
                    noise_window_end_time = Pg_theoretical
                else:   # if P-pick
                    # calculate theoretical time between picks
                    distance = arrival_distances_2D[index][
                                   ind_Pg] * 111.195079734632 * 1000  # [distances in degree, so km/degree * m/km = m]
                    v_p = 5300.0  # [m/s]
                    v_s = 3100.0  # [m/s]
                    delta_p_s = distance * (1 / v_s - 1 / v_p)

                    pick = picks_per_trace[ind_Pg]
                    trace_name = pick.time.isoformat().replace('-', '_').replace('.', '_').replace(':', '_')
                    Sg_theoretical = pick.time + delta_p_s
                    trace_start_time = pick.time - window_p_pick[0]
                    trace_end_time = pick.time + window_p_pick[1]

                    signal_window_start_time = pick.time
                    signal_window_end_time = pick.time + delta_p_s

                    noise_window_start_time = pick.time - delta_p_s
                    noise_window_end_time = pick.time

                # get waveform
                try:
                    waveform = client.get_waveforms(
                        pick.waveform_id.network_code,
                        pick.waveform_id.station_code,
                        pick.waveform_id.location_code,
                        pick.waveform_id.channel_code,
                        trace_start_time,
                        trace_end_time)
                    waveform[0].stats.channel = waveform[0].stats.channel[:-1] + 'Z'

                except FDSNNoDataException as e:
                    print('No data available for this trace')
                    continue


                signal = waveform.slice(
                    starttime=signal_window_start_time, endtime=signal_window_end_time
                )
                signal = signal.detrend("constant")
                signal.taper(max_percentage=0.05, type="hann")

                noise = waveform.slice(
                    starttime=noise_window_start_time, endtime=noise_window_end_time
                )
                noise = noise.detrend("constant")
                noise.taper(max_percentage=0.05, type="hann")

                snr = np.amax(np.abs(signal)) / np.amax(np.abs(noise))
                # trace_snr_db = 10 * np.log10(snr)

                if type(ind_Sg) is int:
                    Sg_pick = picks_per_trace[ind_Sg]
                    trace_s_arrival_sample = int(
                        (Sg_pick.time - waveform[0].stats.starttime) *
                        waveform[0].stats.sampling_rate)
                    trace_s_arrival_theoretical = False
                    S_count += 1
                    pick_count += 1
                else:
                    trace_s_arrival_sample = int(
                        (Sg_theoretical - waveform[0].stats.starttime) *
                        waveform[0].stats.sampling_rate)
                    trace_s_arrival_theoretical = True


                if type(ind_Pg) is int:
                    trace_p_arrival_sample = int((pick.time - waveform[0].stats.starttime) *
                                                 waveform[0].stats.sampling_rate)
                    trace_p_arrival_theoretical = False
                    P_count += 1
                    pick_count += 1
                else:
                    trace_p_arrival_sample = int(
                        (Pg_theoretical - waveform[0].stats.starttime) *
                        waveform[0].stats.sampling_rate)
                    trace_p_arrival_theoretical = True

                trace_params = {
                    "trace_name": trace_name,
                    "trace_start_time": waveform[0].stats.starttime.isoformat(),
                    "trace_sampling_rate_hz": waveform[0].stats.sampling_rate,
                    "trace_ntps": waveform[0].stats.npts,
                    "trace_p_arrival_sample": trace_p_arrival_sample,
                    "trace_s_arrival_sample": trace_s_arrival_sample,
                    "trace_p_arrival_theoretical": trace_p_arrival_theoretical,
                    "trace_s_arrival_theoretical": trace_s_arrival_theoretical,
                    "trace_snr": snr,
                    "trace_polarity": pick.polarity,
                    "trace_p_uncertainty_s": pick.time_errors.lower_uncertainty,
                    "station_network_code": pick.waveform_id.network_code,
                    "station_code": pick.waveform_id.station_code,
                    "trace_channel": pick.waveform_id.channel_code,
                    "station_location_code": pick.waveform_id.location_code}

                event_params = {
                    "source_id": ev_cat['ev_id'],
                    "source_magnitude": ev_cat['mag'],
                    "source_size": ev_cat['mag_plot_size'],
                    "source_magnitude_type": ev_cat['mag_type'],
                    "source_type": ev_cat['ev_type'],
                    # "source_type": ev_cat['event_type'],
                    "source_latitude": o.latitude,
                    "source_longitude": o.longitude,
                    "source_depth_m": o.depth,
                    "source_origin_time": o.time.isoformat()}

                path_params = {
                    'path_hyp_distance_m': distance}

                actual_t_start, data, _ = sbu.stream_to_array(
                    waveform,
                    component_order=writer.data_format["component_order"],
                )
                writer.add_trace({**event_params, **trace_params, **path_params}, data)
                # plt.plot(data[0])
                # plt.show()
                # writer.flush_hdf5() # write to disk
                print('total pick count: ' + str(pick_count) + ', P-picks: ' + str(P_count) +
                      ', S-picks: ' + str(S_count))

if __name__ == '__main__':
    main()