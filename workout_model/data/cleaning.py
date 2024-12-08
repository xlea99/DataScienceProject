import pandas as pd
import os
from workout_model.common.paths import paths
from workout_model.util.utilities import fileStandardSortKey, cleanBadCharacters



#region === Initial Cleaning ===

# This function imports the raw gym exercise data into a pandas dataframe.
def readGymExerciseCSV():
    datasetPath = paths["data"] / "gym_exercise_dataset.csv"
    _rawGymExerciseDataFrame = pd.read_csv(datasetPath)
    return _rawGymExerciseDataFrame

# This function simply drops the unneeded columns from the dataframe, divides multi-entry cells into lists, and
# removes columns listed as variations.
def initialClean(gymExerciseDataFrame):
    # Remove unneeded columns.
    gymExerciseDataFrame = gymExerciseDataFrame.drop(columns=["Preparation","Execution","Force","Secondary Muscles",
                                                                "Stabilizer_Muscles","Antagonist_Muscles",
                                                                "Dynamic_Stabilizer_Muscles","parent_id","Difficulty (1-5)"])

    # Standardize string values in all columns
    for column in gymExerciseDataFrame.columns:
        gymExerciseDataFrame[column] = gymExerciseDataFrame[column].apply(
            lambda x: ' '.join(str(x).split()) if pd.notnull(x) else x
        )

    # Clean up bad/invisible characters from all values.
    for column in gymExerciseDataFrame.columns:
        gymExerciseDataFrame[column] = gymExerciseDataFrame[column].apply(cleanBadCharacters)

    # Remove all tuples in which there is a listed variation
    gymExerciseDataFrame = gymExerciseDataFrame[gymExerciseDataFrame["Variation"].str.strip().str.lower() == "no"]
    gymExerciseDataFrame = gymExerciseDataFrame.drop(columns="Variation")

    return gymExerciseDataFrame

# This function standardizes all values within the dataframe that are given in variations to standard. For example,
# if standard is "On", and variations is ["Active", "working", "on", "Not Off"], all values found equivalent to
# variations will be converted to "On".
def standardizeValue(_gymExerciseDataframe,standard : str,variations : (str,list), target_columns: list = None):
    # Ensure variations is a list for consistent processing
    if(isinstance(variations, str)):
        variations = [variations]

    # Normalize case and whitespace for consistency in matching
    variations = [v.strip().lower() for v in variations]
    standard = standard.strip()

    # Helper function to replace variations with the standard
    def replaceVariations(value):
        if isinstance(value, str) and value.strip().lower() in variations:
            return standard
        return value

    # Apply the replacement function to all targeted columns in the DataFrame
    columnsToProcess = target_columns if target_columns else _gymExerciseDataframe.columns
    for column in columnsToProcess:
        _gymExerciseDataframe[column] = _gymExerciseDataframe[column].apply(replaceVariations)

    return _gymExerciseDataframe

# This method deduplicates tuples based on the uniqueness of lookupColumn(s) with a subset of rows, defined by
# variations in the target_columns.
def deduplicateByLookup(gymExerciseDataFrame, variations: (str, list), targetColumn: str, lookupColumns: (str, list)):
    # Ensure variations is a list
    if isinstance(variations, str):
        variations = [variations]

    # Ensure lookupColumns is a list
    if isinstance(lookupColumns, str):
        lookupColumns = [lookupColumns]

    # Normalize variations for consistent matching
    variations = [v.strip().lower() for v in variations]

    # Step 1: Filter rows based on variations in the target_column
    filtered_subset = gymExerciseDataFrame[gymExerciseDataFrame[targetColumn]
                                            .str.strip()
                                            .str.lower()
                                            .isin(variations)]

    # Step 2: Deduplicate the filtered subset based on lookupColumns uniqueness
    # Sort to prioritize a specific variation (optional, can sort by other criteria)
    filtered_subset = filtered_subset.sort_values(by=targetColumn)

    # Drop duplicates, keeping only the first occurrence based on lookupColumns
    deduplicated_subset = filtered_subset.drop_duplicates(subset=lookupColumns)

    # Step 3: Merge the deduplicated subset with the rest of the original dataframe
    remaining_data = gymExerciseDataFrame[~gymExerciseDataFrame.index.isin(filtered_subset.index)]
    result_df = pd.concat([remaining_data, deduplicated_subset])

    return result_df

#endregion === Initial Cleaning ===
#region === Data Removal ===

# This function removes all tuples that contain equipments we don't want to worry about presenting to the user
# (assisted, plyometric)
def removeBadEquipment(gymExerciseDataFrame):
    badEquipmentTypes = ["Assisted","Assisted (Partner)","Assisted Chest Dip","Plyometric"]
    return gymExerciseDataFrame[~gymExerciseDataFrame["Equipment"].isin(badEquipmentTypes)]

#endregion === Data Removal ===
#region === Data Preprocessing ===

# This method standardizes muscle values for further processing, fixing bad commas, spacing, and representing them
# as lists.
def fixMuscleValues(gymExerciseDataFrame):
    # Some muscles were inconsistently stored with commas in them - this helper method corrects it.
    def applyMuscleCorrections(muscle_string):
        muscle_corrections = {"Trapezius, Upper": "Upper Trapezius",
                              "Trapezius, Middle": "Middle Trapezius",
                              "Trapezius, Lower": "Lower Trapezius",
                              "Pectoralis Major, Sternal": "Sternal Pectoralis Major",
                              "Erector Spinae, Cervicis & Capitis Fibers": "Cervicis & Capitis Erector Spinae",
                              "Sternocleidomastoid, Posterior Fibers": "Posterior Fibers Sternocleidomastoid",}

        if pd.isnull(muscle_string):
            # Return as-is if the value is NaN
            return muscle_string

        # Replace problematic substrings using the correction dictionary
        for incorrect, correct in muscle_corrections.items():
            muscle_string = muscle_string.replace(incorrect, correct)

        return muscle_string

    # First, we apply some corrections due to inconsistent comma inclusion in the dataset
    columnsToProcess = ["Target_Muscles", "Synergist_Muscles"]
    for column in columnsToProcess:
        gymExerciseDataFrame[column] = gymExerciseDataFrame[column].apply(
            lambda x: applyMuscleCorrections(x)
        )

    # Now we process our muscle columns, which may contain, multiple values into lists.
    for column in columnsToProcess:
        gymExerciseDataFrame[column] = gymExerciseDataFrame[column].apply(
            lambda x: [item.strip() for item in str(x).split(",") if item.strip()] if pd.notnull(x) else []
        )

    return gymExerciseDataFrame

# This function uses multi-hot encoding to represent the list of all muscles as features, rather than vague lists,
# then removes the original muscles features. Assumes muscle values have been standardized.
# Multi-hot encodes muscles with distinction:
#     1 = Target muscle
#     2 = Synergist muscle
#     0 = None
def multiHotEncodeMuscles(gymExerciseDataFrame):
    # Generate a set of all unique muscles values in ONLY Target_Muscles.
    uniqueMuscles = getUniqueValuesInColumn(gymExerciseDataFrame,columnName="Target_Muscles",isValueList=True)

    # Initialize new columns for each unique muscle
    for muscle in uniqueMuscles:
        gymExerciseDataFrame[muscle] = 0.0  # Default to 0 for all rows

    # Update each muscle column based on primary and secondary muscles
    for idx, row in gymExerciseDataFrame.iterrows():
        targetMuscles = row["Target_Muscles"]
        synergistMuscles = row["Synergist_Muscles"]

        for muscle in uniqueMuscles:
            if muscle in targetMuscles:
                gymExerciseDataFrame.at[idx, muscle] = 1.0  # Primary muscle
            elif muscle in synergistMuscles:
                gymExerciseDataFrame.at[idx, muscle] = 0.5  # Secondary muscle

    # Drop the original muscle columns
    gymExerciseDataFrame = gymExerciseDataFrame.drop(columns=["Target_Muscles", "Synergist_Muscles"])

    return gymExerciseDataFrame

#endregion === Data Preprocessing ===
#region === Reports ===

# This function generates a list of all unique values in the given column. isValueList accommodates for when the values
# of a column are in list form, rather than string.
def getUniqueValuesInColumn(gymExerciseDataFrame,columnName,isValueList=False):
    uniqueValues = set()
    for idx, row in gymExerciseDataFrame.iterrows():
        value = row[columnName]
        if(isValueList):
            for subValue in value:
                uniqueValues.add(subValue)
        else:
            uniqueValues.add(value)
    return uniqueValues

# This function identifies unique values across all columns, including list columns, to detect unique values and
# inconsistencies manually.
def generateUniqueValueReport(gymExerciseDataFrame,blacklistColumns = None):
    if(not blacklistColumns):
        blacklistColumns = []

    # Make path if missing
    uniquenessReportsPath = paths["reports"] / "uniqueness_reports"
    if(not os.path.exists(uniquenessReportsPath)):
        os.mkdir(uniquenessReportsPath)

    # Loop through and report on all columns
    for column in gymExerciseDataFrame.columns:
        if(column in blacklistColumns):
            continue
        # Temporarily convert list-like columns to strings for processing
        tempColumn = gymExerciseDataFrame[column].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
        # Generate the sorted list of unique values
        uniqueValues = sorted(tempColumn.unique(), key=fileStandardSortKey)

        # Print the column name and unique values
        reportString = f"{column}\n\n"
        for uniqueValue in uniqueValues:
            reportString += f"{uniqueValue}\n"
        with(open(uniquenessReportsPath / f"{column}.txt","w",encoding="utf-8") as f):
            f.write(reportString)

#endregion === Reports ===



#region === Process ===

# Main function to read and pre-process the data to a ready state.
def fullProcessData():
    gymExerciseDataFrame = readGymExerciseCSV()
    gymExerciseDataFrame = initialClean(gymExerciseDataFrame)

    gymExerciseDataFrame = removeBadEquipment(gymExerciseDataFrame)
    gymExerciseDataFrame = fixMuscleValues(gymExerciseDataFrame)
    uniqueMuscles = getUniqueValuesInColumn(gymExerciseDataFrame, columnName="Target_Muscles", isValueList=True)
    gymExerciseDataFrame = multiHotEncodeMuscles(gymExerciseDataFrame)

    # Here, we use deduplication to remove (as defined by our research) distinction-less exercise tuples from the data.
    # We reference "grouped_exercises.txt", which is a document we prepared using generated uniqueness reports to group
    # reasonably similar exercises together, and then let our algorithm detect which tuples are worth preserving and
    # which can be dropped.

    groupedExercises = []
    with open(paths["data"] / "grouped_exercises.txt","r") as f:
        for line in f:
            groupedExercises.append([thisExercise.strip() for thisExercise in line.split(",")])
    for groupedExercise in groupedExercises:
        gymExerciseDataFrame = deduplicateByLookup(gymExerciseDataFrame,variations=groupedExercise,
                                                   targetColumn="Exercise Name",lookupColumns=["Equipment"] + list(uniqueMuscles))

    # Now, we run a second batch of deduplication to remove "ambiguous" exercises, where the name and equipment are
    # identical, yet muscle groups are different.
    uniqueExercises = getUniqueValuesInColumn(gymExerciseDataFrame, columnName="Exercise Name", isValueList=False)
    for uniqueExercise in uniqueExercises:
        gymExerciseDataFrame = deduplicateByLookup(gymExerciseDataFrame,variations=uniqueExercise,
                                                   targetColumn="Exercise Name",lookupColumns="Equipment")


    generateUniqueValueReport(gymExerciseDataFrame,blacklistColumns=uniqueMuscles)
    return gymExerciseDataFrame

#endregion === Process ===

df = fullProcessData()
print(df.head(20).to_string())
print(df[df["Exercise Name"].isin(["Leg Presses: 45° Leg Press"])].to_string())