import random
from datetime import timedelta

import pandas as pd

# calculate the gap different between the 'Start' column value of the row and the 'Start' column value of the
# previous row with the same 'Equipment' value that was with 'Maintenance Type' of scheduled if there is no
# prev_schedule_row its put -1 . if I do df['Start'] <= row['Start'] the 2 rows with the same time and date and one
# is scheduled and one damage calculate will be 0.


def calculate_hours_from_last_maintenance(df: pd.DataFrame) -> pd.DataFrame:
    # Sort the DataFrame by 'Equipment' and 'Start' columns
    df = df.sort_values(by=['Equipment', 'Start'])

    # Initialize a new column to store the time differences
    df['hours_from_last_schedule_maintenance'] = None

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Find the previous row with the same 'Equipment' value and 'Maintenance Type' of 'scheduled'
        prev_schedule_rows = df[(df['Equipment'] == row['Equipment']) &
                                (df['Maintenance Type'] == 'Scheduled') &
                                (df['Start'] < row['Start'])]
        if not prev_schedule_rows.empty:
            prev_schedule_row = prev_schedule_rows.iloc[-1]
            # Calculate the time difference in hours
            time_diff_hours = (row['Start'] - prev_schedule_row['Start']).total_seconds() / 3600
        else:
            # Set time difference to -1 if there is no previous row with maintenance type 'scheduled'
            time_diff_hours = -1
        # Update the value in the new column
        df.at[index, 'hours_from_last_schedule_maintenance'] = time_diff_hours
    return df


# /////////////////////AREA/////////////////////////////////////////

def perform_one_hot_encoding(df, column_name):

    # Perform one-hot encoding on the specified column
    dummies = pd.get_dummies(df[column_name], prefix=column_name).astype(int)
    # Concatenate the one-hot encoded features with the original DataFrame
    df = pd.concat([df, dummies], axis=1)
    # Drop the original column
    df.drop(column_name, axis=1, inplace=True)
    return df


# Calculate damage counts for different window sizes and concatenate the results to the DataFrame.
def calculate_damage_count(df):
    # Filter the DataFrame based on conditions
    filtered_df = df[(~df['hour'].isin([6, 18])) & (df['ServiceGroup'] == '1') & (~df['Maintenance Type'].isin(['Scheduled']))]
    grouped = filtered_df.groupby('Equipment')

    Damage_Count = []
    for window_size in [30, 60, 90]:
        window_name = f'{window_size}D'
        counts = []
        for name, group in grouped:
            group = group.set_index('Start')  # Set the index to 'Start' for easier date-based operations
            group[f'Damage_Count{window_name}'] = group.index.to_series().rolling(window_name).count() - 1
            counts.append(group.reset_index())
        Damage_Count.append({window_name: pd.concat(counts)})
    df_all = Damage_Count[0].get('30D')
    df_all['Damage_Count60D'] = Damage_Count[1].get('60D')['Damage_Count60D']
    df_all['Damage_Count90D'] = Damage_Count[2].get('90D')['Damage_Count90D']
    df_all = df_all.iloc[:, [1, 0] + list(range(2, len(df_all.columns)))]
    return df_all


def drop_column(df):
    df.drop('NotificationCode', axis=1, inplace=True)


def determine_service_group(df):
    distinct_equipment = df['Equipment'].unique()
    distinct_equipment_df = pd.DataFrame(distinct_equipment, columns=['Equipment'])

    # Filter scheduled maintenance records based on the specified hours
    scheduled_maintenance_records = df[
        (df['Maintenance Type'] == 'Scheduled') &
        (df['hour'].isin([0, 6, 12, 18]))
    ]

    def determine_group(equipment_name):
        # Filter records for the current equipment and check for maintenance at hours 6 or 18
        equipment_records = scheduled_maintenance_records[
            (scheduled_maintenance_records['Equipment'] == equipment_name) &
            (scheduled_maintenance_records['hour'].isin([6, 18]))
        ]

        # If there are any records for these hours, it's Group 1, else it's Group 2
        if not equipment_records.empty:
            return '1'
        else:
            return '2'

    # Apply the function to each piece of equipment to determine its service group
    distinct_equipment_df['ServiceGroup'] = distinct_equipment_df['Equipment'].apply(determine_group)

    # Map the service group to each piece of equipment in the main DataFrame
    group_1_equipment_list = distinct_equipment_df[distinct_equipment_df['ServiceGroup'] == '1']['Equipment'].tolist()
    equipment_service_group_mapping = {equipment: ('1' if equipment in group_1_equipment_list else '2') for equipment in df['Equipment'].unique()}
    df['ServiceGroup'] = df['Equipment'].map(equipment_service_group_mapping)

    return df

# Add a new column 'Label' to the DataFrame with all rows set to 1.
def add_label_column(df):
    df['Label'] = 1
    return df


def generate_new_rows_between_consecutive(df):
    """
    Generate new rows between consecutive rows in the DataFrame where the time difference
    between them is greater than 12 hours.

    Args:
    - df (pandas.DataFrame): Input DataFrame containing the service treatments data.

    Returns:
    - pandas.DataFrame: Updated DataFrame with new rows inserted between consecutive rows.
    """



    # Sort DataFrame by 'Equipment' and 'Start' columns
    df.sort_values(by=['Equipment', 'Start'], inplace=True)

    # Define a function to generate new rows between consecutive rows
    # Define a function to generate new rows between consecutive rows
    # Define a function to generate new rows between consecutive rows
    def generate_new_rows(row1, row2):
        new_rows = []
        start_time = row1['Start']
        end_time = row2['Start']
        time_diff = end_time - start_time
        prev_maintenance_hours = row1['hours_from_last_schedule_maintenance']  # Get hours from previous row
        while time_diff > timedelta(hours=12):
            random_jump = random.randint(10, 20)  # Random jump between 10 to 20 hours
            new_start_time = start_time + timedelta(hours=random_jump)
            new_row = row1.copy()
            new_row['Start'] = new_start_time
            new_row['Downtime (Hrs)'] = 0
            # Set other columns to 0 or copy from previous row as required
            new_row['Label'] = 0
            if new_row['Label'] != 1:  # Check if label is not 1
                time_difference_hours = (new_start_time - row1['Start']).total_seconds() / 3600
                new_row['hours_from_last_schedule_maintenance'] += time_difference_hours

            new_rows.append(new_row)
            start_time = new_start_time
            time_diff = end_time - start_time
        return pd.DataFrame(new_rows)

    # Create a list of DataFrames for each group
    new_dfs = []
    prev_row = None
    for index, row in df.iterrows():
        if prev_row is not None and index != len(df) - 1:  # Skip calculation for the last row
            new_dfs.append(generate_new_rows(prev_row, row))
        prev_row = row

    # Concatenate all DataFrames in the list
    if new_dfs:
        new_df = pd.concat(new_dfs, ignore_index=True)
        # Concatenate the original DataFrame with the new DataFrame
        updated_df = pd.concat([df, new_df], ignore_index=True)
    else:
        # If no new rows were generated, return the original DataFrame
        updated_df = df.copy()

    # Sort DataFrame by 'Equipment' and 'Start' columns again
    updated_df.sort_values(by=['Equipment', 'Start'], inplace=True)

    return updated_df


def calculate_damage_counts_df(df):

    # Sort DataFrame by 'Equipment' and 'Start' columns
    df.sort_values(by=['Equipment', 'Start'], inplace=True)

    # Define a function to calculate damage counts for each row
    def calculate_damage_counts(row, days):
        previous_rows = df[(df['Equipment'] == row['Equipment']) &
                           (df['Start'] < row['Start']) &
                           (row['Start'] - df['Start'] <= timedelta(days=days)) &
                           (df['Downtime (Hrs)'] > 0)]
        return previous_rows.shape[0]

    # Calculate Damage_Count30D, Damage_Count60D, and Damage_Count90D
    df['Damage_Count30D'] = df.apply(lambda row: calculate_damage_counts(row, 30), axis=1)
    df['Damage_Count60D'] = df.apply(lambda row: calculate_damage_counts(row, 60), axis=1)
    df['Damage_Count90D'] = df.apply(lambda row: calculate_damage_counts(row, 90), axis=1)

    return df


# Read the CSV file into a pandas DataFrame
df = pd.read_csv('/Users/oriyasheetrit/PycharmProjects/FinalProject/All Events Barminco_new1.csv')
# Convert 'Start' and 'DateTimeFinish' columns to datetime objects
df['Start'] = pd.to_datetime(df['Start'], format='%d.%m.%Y %H:%M')
df['DateTimeFinish'] = pd.to_datetime(df['DateTimeFinish'], format='%d.%m.%Y %H:%M')
# Convert 'Start' column to datetime format
df['Start'] = pd.to_datetime(df['Start'])


df = calculate_hours_from_last_maintenance(df)
df = determine_service_group(df)
df = filtered_df = df[(~df['hour'].isin([6, 18])) & (df['ServiceGroup'] == '1') & (~df['Maintenance Type'].isin(['Scheduled']))]
# df = calculate_damage_count(df)
drop_column(df)
df = perform_one_hot_encoding(df, 'Area')
df = add_label_column(df)
df.drop("DateTimeFinish", axis=1, inplace=True)
df.drop("Notification Description", axis=1, inplace=True)
df.drop("Maintenance Type", axis=1, inplace=True)
df.drop("Downtime", axis=1, inplace=True)
df.drop("Component", axis=1, inplace=True)
df.drop("year", axis=1, inplace=True)
df.drop("month", axis=1, inplace=True)
df.drop("day", axis=1, inplace=True)
df.drop("ServiceGroup", axis=1, inplace=True)
df.drop("hour", axis=1, inplace=True)
df = generate_new_rows_between_consecutive(df)
df = calculate_damage_counts_df(df)
df.drop("Downtime (Hrs)", axis=1, inplace=True)
# df.show()
df.to_csv('model.csv')