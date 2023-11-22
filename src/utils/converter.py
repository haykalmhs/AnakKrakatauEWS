from obspy.core.util.attribdict import AttribDict
from obspy.core.inventory import Inventory, read_inventory
from infrapy.detection import beamforming_new
from pathlib import Path
import pandas as pd
from obspy import read
from infrapy.utils.data_io import json_to_detection_list
from tqdm.notebook import tqdm
import numpy as np
import json
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


def sac_to_csv(file_path, output_folder, freq):
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
    stream.filter('lowpass', freq=freq)

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
    df_output.to_csv(csv_output_path)

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
        grouped_path = f"{sac_file.parent}/{network}.I06H*.{channel}.{times}.sac"
        grouped_paths[key] = grouped_path

    # Create a DataFrame from the grouped paths
    df_grouped_paths = pd.DataFrame(list(grouped_paths.values()), columns=['GroupedFilePath'])
    
    return df_grouped_paths


def extract_det_from_csv(csv_folder, detection_file, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the detection file
    detections = json_to_detection_list(detection_file)

    # Track processed detections
    processed_detections = set()

    # Get a list of all CSV files in the folder
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # Process each CSV file with progress bar
    for file in tqdm(csv_files, desc="Processing CSV files"):
        file_path = os.path.join(csv_folder, file)

        # Read the CSV file, assuming the first column contains the timestamp
        df = pd.read_csv(file_path, index_col=0)
        df.index = pd.to_datetime(df.index)  # Convert the index to datetime

        # Iterate over each detection with progress bar
        for i, det in tqdm(enumerate(detections), total=len(detections), desc=f"Processing detections for {file}", leave=False):
            # Skip detection if it's already processed
            if i in processed_detections:
                continue

            # Define start and end times for slicing
            t_start = pd.to_datetime(det.peakF_UTCtime) + pd.to_timedelta(det.start, 's')
            t_end = pd.to_datetime(det.peakF_UTCtime) + pd.to_timedelta(det.end, 's')

            # Filter the DataFrame for the given detection time range
            trimmed_df = df[(df.index >= t_start) & (df.index <= t_end)]

            # Skip if the trimmed DataFrame is empty
            if trimmed_df.empty:
                continue

            # Mark the detection as processed
            processed_detections.add(i)

            # Format times for the output filename
            t_start_str = t_start.strftime('%Y%m%d_%H%M%S')
            t_end_str = t_end.strftime('%Y%m%d_%H%M%S')

            # Save the trimmed DataFrame to a new CSV file in the output folder
            trimmed_filename = os.path.join(output_folder, f"{file.split('.')[0]}_det_{i}_{t_start_str}_to_{t_end_str}.csv")
            trimmed_df.to_csv(trimmed_filename, index=True)
            print(f"Saved trimmed data to {trimmed_filename}")

