from obspy.core.util.attribdict import AttribDict
from obspy.core.inventory import Inventory, read_inventory
from infrapy.detection import beamforming_new
from pathlib import Path
import pandas as pd
from obspy import read
import re
import glob
import os

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', '_', filename)

def mseed_to_sac(input_directory, output_directory, stationxml_directory):
    # Check if output directory exists, create it if it doesn't
    # List all mseed files in the input directory
    mseed_files = [f for f in os.listdir(input_directory) if f.endswith('.mseed')]

    # Process each mseed file
    for mseed_file in mseed_files:
        # Extract the station code from the mseed filename
        # This assumes the station code is the part of the filename before the first dot
        station_code = mseed_file.split('.')[1]  # Adjust based on your file naming convention

        # Construct the path to the corresponding StationXML file
        stationxml_file = os.path.join(stationxml_directory, f"IM.{station_code}.xml")

        # Check if the corresponding StationXML file exists
        if not os.path.isfile(stationxml_file):
            print(f"StationXML file for {station_code} does not exist. Skipping {mseed_file}.")
            continue

        # Read the station metadata from the StationXML file
        inventory = read_inventory(stationxml_file)

        # Read the mseed file into a Stream object
        st = read(os.path.join(input_directory, mseed_file))
        
        # Enrich Stream with metadata from StationXML
        st.attach_response(inventory)

        # Loop through each Trace in the Stream
        for tr in st:
            # Attach station metadata to trace stats
            tr.stats.sac = {}
            for net in inventory:
                for sta in net:
                    if tr.stats.station == sta.code:
                        tr.stats.sac.stla = sta.latitude
                        tr.stats.sac.stlo = sta.longitude
                        tr.stats.sac.stel = sta.elevation
                        break

            # Use the original mseed filename with the .sac extension
            sac_filename = os.path.splitext(mseed_file)[0] + '.sac'
            # Write the Trace as a SAC file in the output directory
            tr.write(os.path.join(output_directory, sac_filename), format='SAC')


def sac_to_csv(file_path, output_folder, freqmin, freqmax):
    # Ensure the output directory exists
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    # Extract the filename from the input file path and sanitize it
    file_name = os.path.basename(file_path)
    sanitized_file_name = sanitize_filename(file_name)

    # Remove wildcard characters and replace them with an underscore
    sanitized_file_name = sanitized_file_name.replace('*', '_')

    # Read SAC file into a stream
    stream = read(file_path)

    # Apply the Butterworth bandpass filter with configurable freqmin and freqmax
    stream.filter('bandpass', freqmin=freqmin, freqmax=freqmax)

    # Process the stream using the custom function provided
    x, t, t0, _ = beamforming_new.stream_to_array_data(stream)

    # Extract station names from SAC file headers
    station_names = [tr.stats.station for tr in stream]

    # Create a DataFrame from the time series data with station headers
    df_output = pd.DataFrame(x.T, index=(pd.to_datetime(t0) + pd.to_timedelta(t, unit='s')), columns=station_names)

    # Define the output CSV file path, ensuring that the file extension is .csv
    csv_file_name = f'{os.path.splitext(sanitized_file_name)[0]}.csv'
    csv_output_path = os.path.join(output_folder, csv_file_name)

    # Write the DataFrame to a CSV file
    df_output.to_csv(csv_output_path, index_label='DateTime')

    return csv_output_path  # Optionally return the path to the created CSV file
    
def sac_file_parser(filepath):
    filename = os.path.basename(filepath)
    parts = filename.split('.')
    network = parts[0]
    station = parts[1]
    channel = parts[2]
    times = ''.join(parts[3:-1])  # Exclude the extension part
    return network, station, channel, times

def sac_path_grouping(folder_path):
    # Get a list of all .sac files in the folder
    sac_files = list(Path(folder_path).rglob('*.sac'))

    # Dictionary to hold the grouped paths
    grouped_paths = {}

    # Parse and group the file paths
    for sac_file in sac_files:
        network, station, channel, times = sac_file_parser(sac_file.name)
        key = (network, channel, times)
        grouped_path = f"{network}.I06H*.{channel}.{times}.sac"
        grouped_paths[key] = grouped_path

    # Create a DataFrame from the grouped paths
    df_grouped_paths = pd.DataFrame(list(grouped_paths.values()), columns=['GroupedFilePath'])
    
    return df_grouped_paths
