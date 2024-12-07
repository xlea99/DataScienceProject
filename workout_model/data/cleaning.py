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

# This function simply drops the unneeded columns from the dataframe, and divides multi-entry cells into lists.
def prepareColumns(_gymExerciseDataFrame):
    # Remove unneeded columns.
    _gymExerciseDataFrame = _gymExerciseDataFrame.drop(columns=["Preparation","Execution","Force","Synergist_Muscles",
                                                                "Stabilizer_Muscles","Antagonist_Muscles",
                                                                "Dynamic_Stabilizer_Muscles","parent_id","Difficulty (1-5)"])

    # Standardize string values in all columns
    for column in _gymExerciseDataFrame.columns:
        _gymExerciseDataFrame[column] = _gymExerciseDataFrame[column].apply(
            lambda x: ', '.join(item.strip() for item in str(x).split(',')) if pd.notnull(x) else x
        )

    # Clean up bad/invisible characters from all values.
    for column in _gymExerciseDataFrame.columns:
        _gymExerciseDataFrame[column] = _gymExerciseDataFrame[column].apply(cleanBadCharacters)

    return _gymExerciseDataFrame

# This function standardizes all values within the dataframe that are given in variations to standard. For example,
# if standard is "On", and variations is ["Active", "working", "on", "Not Off"], all values found equivalent to
# variations will be converted to "On"
def standardizeValue(_gymExerciseDataFrame,standard : str,variations : (str,list), target_columns: list = None):
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
    columnsToProcess = target_columns if target_columns else _gymExerciseDataFrame.columns
    for column in columnsToProcess:
        _gymExerciseDataFrame[column] = _gymExerciseDataFrame[column].apply(replaceVariations)

    return _gymExerciseDataFrame

#endregion === Initial Cleaning ===

#region === Data Removal ===

# This function removes all tuples that contain equipments we don't want to worry about presenting to the user
# (assisted, plyometric)
def removeBadEquipment(_gymExerciseDataFrame):
    badEquipmentTypes = ["Assisted","Assisted (Partner)","Assisted Chest Dip","Plyometric"]
    return _gymExerciseDataFrame[~_gymExerciseDataFrame["Equipment"].isin(badEquipmentTypes)]

#endregion === Data Removal ===

#region === Data Preprocessing ===

# This function uses multi-hot encoding to represent the list of all muscles as features, rather than vague lists,
# then removes the original muscles features.
# Multi-hot encodes muscles with distinction:
#     1 = Target muscle
#     2 = Secondary muscle
#     0 = None
def multiHotEncodeMuscles(_gymExerciseDataFrame):
    # First, we process our muscle columns, which may contain, multiple values into lists.
    columnsToSplit = ["Target_Muscles", "Secondary Muscles"]
    for column in columnsToSplit:
        _gymExerciseDataFrame[column] = _gymExerciseDataFrame[column].apply(
            lambda x: [item.strip() for item in str(x).split(",") if item.strip()] if pd.notnull(x) else [])

    # Generate a set of all unique muscles values between both muscle columns.
    uniqueMuscles = set()
    for idx, row in _gymExerciseDataFrame.iterrows():
        targetMuscles = row["Target_Muscles"]
        secondaryMuscles = row["Secondary Muscles"]

        for targetMuscle in targetMuscles:
            uniqueMuscles.add(targetMuscle)
        for secondaryMuscle in secondaryMuscles:
            uniqueMuscles.add(secondaryMuscle)

    # Initialize new columns for each unique muscle
    for muscle in uniqueMuscles:
        _gymExerciseDataFrame[muscle] = 0  # Default to 0 for all rows

    # Update each muscle column based on primary and secondary muscles
    for idx, row in _gymExerciseDataFrame.iterrows():
        targetMuscles = row["Target_Muscles"]
        secondaryMuscles = row["Secondary Muscles"]

        for muscle in uniqueMuscles:
            if muscle in targetMuscles:
                _gymExerciseDataFrame.at[idx, muscle] = 1  # Primary muscle
            elif muscle in secondaryMuscles:
                _gymExerciseDataFrame.at[idx, muscle] = 2  # Secondary muscle

    # Drop the original muscle columns
    _gymExerciseDataFrame = _gymExerciseDataFrame.drop(columns=["Target_Muscles", "Secondary Muscles"])

    return _gymExerciseDataFrame


#endregion === Data Preprocessing ===

#region === Reports ===

# This function identifies unique values across all columns, including list columns, to detect inconsistencies.
def generateInconsistentValueReport(_gymExerciseDataFrame):
    # Make path if missing
    inconsistencyReportsPath = paths["reports"] / "inconsistency_reports"
    if(not os.path.exists(inconsistencyReportsPath)):
        os.mkdir(inconsistencyReportsPath)

    # Loop through and report on all columns
    for column in _gymExerciseDataFrame.columns:
        # Temporarily convert list-like columns to strings for processing
        tempColumn = _gymExerciseDataFrame[column].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else x
        )
        # Generate the sorted list of unique values
        uniqueValues = sorted(tempColumn.unique(), key=fileStandardSortKey)

        # Print the column name and unique values
        reportString = f"{column}\n\n"
        for uniqueValue in uniqueValues:
            reportString += f"{uniqueValue}\n"
        with(open(inconsistencyReportsPath / f"{column}.txt","w",encoding="utf-8") as f):
            f.write(reportString)

#endregion === Reports ===

gymExerciseDataFrame = readGymExerciseCSV()
gymExerciseDataFrame = prepareColumns(gymExerciseDataFrame)
gymExerciseDataFrame = standardizeValue(gymExerciseDataFrame,standard="Neck Flexion",variations=["Neck Flexion",
                                                                                                   "Lateral Neck Flexion"],
                                        target_columns=["Exercise Name"])
gymExerciseDataFrame = removeBadEquipment(gymExerciseDataFrame)


generateInconsistentValueReport(gymExerciseDataFrame)
print("==========================================")
print("==========================================")
print("==========================================")
gymExerciseDataFrame = multiHotEncodeMuscles(gymExerciseDataFrame)
print(gymExerciseDataFrame.head(20).to_string())