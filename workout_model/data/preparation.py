import pandas as pd
import os
import json
from workout_model.common.paths import paths
from workout_model.util.utilities import fileStandardSortKey, cleanBadCharacters

#region === Initial Cleaning ===

# This function imports the raw gym exercise data into a pandas dataframe.
def readGymExerciseCSV():
    datasetPath = paths["data"] / "gym_exercise_dataset.csv"
    _rawexerciseDataFrame = pd.read_csv(datasetPath)
    return _rawexerciseDataFrame

# This function simply drops the unneeded columns from the dataframe, divides multi-entry cells into lists, and
# removes columns listed as variations.
def initialClean(exerciseDataFrame):
    # Remove unneeded columns.
    exerciseDataFrame = exerciseDataFrame.drop(columns=["Preparation","Execution","Force","Secondary Muscles",
                                                                "Stabilizer_Muscles","Antagonist_Muscles","Variation",
                                                                "Dynamic_Stabilizer_Muscles","parent_id",
                                                                "Difficulty (1-5)","Utility"])

    # Standardize string values in all columns
    for column in exerciseDataFrame.columns:
        exerciseDataFrame[column] = exerciseDataFrame[column].apply(
            lambda x: ' '.join(str(x).split()) if pd.notnull(x) else x
        )

    # Clean up bad/invisible characters from all values.
    for column in exerciseDataFrame.columns:
        exerciseDataFrame[column] = exerciseDataFrame[column].apply(cleanBadCharacters)

    return exerciseDataFrame

# This function standardizes all values within the dataframe that are given in variations to standard. For example,
# if standard is "On", and variations is ["Active", "working", "on", "Not Off"], all values found equivalent to
# variations will be converted to "On".
def standardizeValue(_exerciseDataFrame,standard : str,variations : (str,list), target_columns: list = None):
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
    columnsToProcess = target_columns if target_columns else _exerciseDataFrame.columns
    for column in columnsToProcess:
        _exerciseDataFrame[column] = _exerciseDataFrame[column].apply(replaceVariations)

    return _exerciseDataFrame

# This method deduplicates tuples based on the uniqueness of lookupColumn(s) with a subset of rows, defined by
# variations in the target_columns.
def deduplicateByLookup(exerciseDataFrame, variations: (str, list), targetColumn: str, lookupColumns: (str, list)):
    # Ensure variations is a list
    if isinstance(variations, str):
        variations = [variations]

    # Ensure lookupColumns is a list
    if isinstance(lookupColumns, str):
        lookupColumns = [lookupColumns]

    # Normalize variations for consistent matching
    variations = [v.strip().lower() for v in variations]

    # Step 1: Filter rows based on variations in the target_column
    filtered_subset = exerciseDataFrame[exerciseDataFrame[targetColumn]
                                            .str.strip()
                                            .str.lower()
                                            .isin(variations)]

    # Step 2: Deduplicate the filtered subset based on lookupColumns uniqueness
    # Sort to prioritize a specific variation (optional, can sort by other criteria)
    filtered_subset = filtered_subset.sort_values(by=targetColumn)

    # Drop duplicates, keeping only the first occurrence based on lookupColumns
    deduplicated_subset = filtered_subset.drop_duplicates(subset=lookupColumns)

    # Step 3: Merge the deduplicated subset with the rest of the original dataframe
    remaining_data = exerciseDataFrame[~exerciseDataFrame.index.isin(filtered_subset.index)]
    result_df = pd.concat([remaining_data, deduplicated_subset])

    return result_df

#endregion === Initial Cleaning ===
#region === Data Removal ===

# This function removes all tuples that contain equipments we don't want to worry about presenting to the user
# (assisted, plyometric)
def removeBadEquipment(exerciseDataFrame):
    badEquipmentTypes = ["Assisted","Assisted (Partner)","Assisted Chest Dip","Plyometric"]
    return exerciseDataFrame[~exerciseDataFrame["Equipment"].isin(badEquipmentTypes)]

#endregion === Data Removal ===
#region === Data Preprocessing ===

# This method standardizes muscle values for further processing, fixing bad commas, spacing, and representing them
# as lists.
def fixMuscleValues(exerciseDataFrame):
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
    columnsToProcess = ["Target_Muscles"]
    for column in columnsToProcess:
        exerciseDataFrame[column] = exerciseDataFrame[column].apply(
            lambda x: applyMuscleCorrections(x)
        )

    # Now we process our muscle columns, which may contain multiple values into lists.
    for column in columnsToProcess:
        exerciseDataFrame[column] = exerciseDataFrame[column].apply(
            lambda x: [item.strip() for item in str(x).split(",") if item.strip()] if pd.notnull(x) else []
        )

    return exerciseDataFrame

# This function uses multi-hot encoding to represent the list of all muscles as features, rather than vague lists,
# then removes the original muscles features. Assumes muscle values have been fixed by fixMusclesValues.
# Multi-hot encodes muscles with distinction:
#     1 = Target muscle
#     0 = None
def multiHotEncodeMuscles(exerciseDataFrame):
    # Generate a set of all unique muscles values in ONLY Target_Muscles.
    uniqueMuscles = getUniqueValuesInColumn(exerciseDataFrame,columnName="Target_Muscles",isValueList=True)

    # Initialize new columns for each unique muscle
    for muscle in uniqueMuscles:
        exerciseDataFrame[muscle] = 0.0  # Default to 0 for all rows

    # Update each muscle column based on primary and secondary muscles
    for idx, row in exerciseDataFrame.iterrows():
        targetMuscles = row["Target_Muscles"]
        synergistMuscles = row["Synergist_Muscles"]

        for muscle in uniqueMuscles:
            if muscle in targetMuscles:
                exerciseDataFrame.at[idx, muscle] = 1.0  # Primary muscle
            elif muscle in synergistMuscles:
                exerciseDataFrame.at[idx, muscle] = 0.5  # Secondary muscle

    # Drop the original muscle columns
    exerciseDataFrame = exerciseDataFrame.drop(columns=["Target_Muscles", "Synergist_Muscles"])

    return exerciseDataFrame

# This function uses multi-hot encoding to represent the 'Equipment' column, creating a new column for each
# unique equipment type. Removes the original 'Equipment' column after encoding, and merges all relevant duplicate
# tuples.
def multiHotEncodeAndMergeEquipment(exerciseDataFrame):
    # Generate a set of all unique equipment values
    uniqueEquipment = exerciseDataFrame["Equipment"].unique()

    # Initialize new columns for each unique equipment type
    for equipment in uniqueEquipment:
        # Default to 0 for all rows
        exerciseDataFrame[equipment] = 0

    # Update the equipment columns with 1.0 where the equipment matches
    for idx, row in exerciseDataFrame.iterrows():
        equipment = row["Equipment"]
        if equipment in uniqueEquipment:
            exerciseDataFrame.at[idx, equipment] = 1

    # Drop the original 'Equipment' column after encoding
    exerciseDataFrame = exerciseDataFrame.drop(columns=["Equipment"])

    # Merge rows that are identical except for equipment columns
    # Identify columns that are not equipment columns (all other columns)
    nonEquipmentColumns = [col for col in exerciseDataFrame.columns if col not in uniqueEquipment]

    # Group by non-equipment columns and aggregate equipment columns
    exerciseDataFrame = (
        exerciseDataFrame.groupby(nonEquipmentColumns, as_index=False)
        .sum()  # Sum the equipment columns to merge the multi-hot encodings
    )

    return exerciseDataFrame

# This function applies manual exercise renaming as found in manual_exercise_renames.csv
def applyManualExerciseRenaming(exerciseDataFrame):
    datasetPath = paths["data"] / "manual_exercise_renames.csv"
    manualExerciseRenaming = pd.read_csv(datasetPath)

    # Filter manualExerciseRenaming to only rows where "NEW Name" is set (not NaN or empty)
    manualExerciseRenaming = manualExerciseRenaming.dropna(subset=["NEW Name"])

    # Merge the manual renaming dataframe into exerciseDataFrame on common columns
    # We exclude the "NEW Name" column during the merge to avoid overwriting data
    mergeColumns = [col for col in manualExerciseRenaming.columns if col != "NEW Name"]
    exerciseDataFrame = exerciseDataFrame.merge(
        manualExerciseRenaming[["NEW Name"] + mergeColumns],
        on=mergeColumns,
        how="left"
    )

    # Apply the renamings: If "NEW Name" is not null, replace the "Exercise Name" column
    exerciseDataFrame["Exercise Name"] = exerciseDataFrame["NEW Name"].combine_first(exerciseDataFrame["Exercise Name"])

    # Drop the "NEW Name" column as it's no longer needed
    exerciseDataFrame = exerciseDataFrame.drop(columns=["NEW Name"])

    return exerciseDataFrame



#endregion === Data Preprocessing ===
#region === Data Enrichment ===

# Adds in values (from research) on both the equipment category, and the setup times associated with them.
def addEquipmentGroupsAndTimes(equipmentDataFrame):
    # First, read the equipment_groups.json file
    with open(paths["data"] / "equipment_groups.json","r") as f:
        equipmentGroups = json.load(f)

    # Add columns for the group name and setup time
    equipmentDataFrame["Equipment Group"] = None
    equipmentDataFrame["Setup Time"] = None

    # Iterate through each group and its entries
    for equipmentGroup in equipmentGroups["groups"]:
        groupName = equipmentGroup["name"]
        setupTime = equipmentGroup["setupTime"]

        for equipmentType in equipmentGroup["entries"]:
            # Find rows matching the current equipmentType and update their group and setup time
            equipmentDataFrame.loc[equipmentDataFrame["Equipment"] == equipmentType, "Equipment Group"] = groupName
            equipmentDataFrame.loc[equipmentDataFrame["Equipment"] == equipmentType, "Setup Time"] = setupTime

    return equipmentDataFrame

# Adds in values (from research) for muscle groups, based on the individual muscle.
def addMuscleGroups(muscleDataFrame):
    # First, read the equipment_groups.json file
    with open(paths["data"] / "muscle_groups.json", "r") as f:
        muscleGroups = json.load(f)

    # Add columns for the group name and setup time
    muscleDataFrame["Muscle Group"] = None

    # Iterate through each group and its entries
    for muscleGroup in muscleGroups["groups"]:
        groupName = muscleGroup["name"]

        for equipmentType in muscleGroup["entries"]:
            # Find rows matching the current equipmentType and update their group and setup time
            muscleDataFrame.loc[muscleDataFrame["Muscle"] == equipmentType, "Muscle Group"] = groupName

    return muscleDataFrame

#endregion === Data Enrichment ===
#region === Reports ===

# This function generates a list of all unique values in the given column. isValueList accommodates for when the values
# of a column are in list form, rather than string.
def getUniqueValuesInColumn(exerciseDataFrame,columnName,isValueList=False):
    uniqueValues = set()
    for idx, row in exerciseDataFrame.iterrows():
        value = row[columnName]
        if(isValueList):
            for subValue in value:
                uniqueValues.add(subValue)
        else:
            uniqueValues.add(value)
    return uniqueValues

# This function identifies unique values across all columns, including list columns, to detect unique values and
# inconsistencies manually.
def generateUniqueValueReport(exerciseDataFrame,blacklistColumns = None):
    if(not blacklistColumns):
        blacklistColumns = []

    # Make path if missing
    uniquenessReportsPath = paths["reports"] / "uniqueness_reports"
    if(not os.path.exists(uniquenessReportsPath)):
        os.mkdir(uniquenessReportsPath)

    # Loop through and report on all columns
    for column in exerciseDataFrame.columns:
        if(column in blacklistColumns):
            continue
        # Temporarily convert list-like columns to strings for processing
        tempColumn = exerciseDataFrame[column].apply(
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

# This function generates a report of remaining duplicate tuples that ARE meaningful, in that despite having the same
# name, they hit different muscle groups and use different exercises, for the purpose of clarity and transparency.
def generateDuplicateExerciseNameReport(exerciseDataFrame, groupColumns):
    # Group by Exercise Name and check for differences across the grouping columns
    grouped = exerciseDataFrame.groupby("Exercise Name")[groupColumns].nunique()

    # Find exercises with more than 1 unique combination in the grouping columns
    duplicates = grouped[grouped.gt(1).any(axis=1)]

    # Filter the original dataframe to only include rows with duplicate Exercise Names
    duplicateExercises = exerciseDataFrame[exerciseDataFrame["Exercise Name"].isin(duplicates.index)]

    with open(paths["reports"] / "duplicateExerciseNameTuples.csv","w") as f:
        f.write(duplicateExercises.to_csv())
    return duplicateExercises

#endregion === Reports ===

#region === Data Processing ===

# Main function to read and pre-process the data to a ready state.
def fullProcessData():

    #region === Exercise Data Frame ===

    exerciseDataFrame = readGymExerciseCSV()
    exerciseDataFrame = initialClean(exerciseDataFrame)

    exerciseDataFrame = removeBadEquipment(exerciseDataFrame)
    exerciseDataFrame = fixMuscleValues(exerciseDataFrame)
    uniqueMuscles = getUniqueValuesInColumn(exerciseDataFrame, columnName="Target_Muscles", isValueList=True)
    exerciseDataFrame = multiHotEncodeMuscles(exerciseDataFrame)

    # Here, we use deduplication to remove (as defined by our research) distinction-less exercise tuples from the data.
    # We reference "grouped_exercises.txt", which is a document we prepared using generated uniqueness reports to group
    # reasonably similar exercises together, and then let our algorithm detect which tuples are worth preserving and
    # which can be dropped.
    groupedExercises = []
    with open(paths["data"] / "grouped_exercises.txt","r") as f:
        for line in f:
            groupedExercises.append([thisExercise.strip() for thisExercise in line.split(",")])
    for groupedExercise in groupedExercises:
        exerciseDataFrame = deduplicateByLookup(exerciseDataFrame,variations=groupedExercise,
                                                   targetColumn="Exercise Name",lookupColumns=["Equipment"] + list(uniqueMuscles))

    # Now, we run a second batch of deduplication to remove "ambiguous" exercises, where the name and equipment are
    # identical, yet muscle groups are different.
    uniqueExercises = getUniqueValuesInColumn(exerciseDataFrame, columnName="Exercise Name", isValueList=False)
    for uniqueExercise in uniqueExercises:
        exerciseDataFrame = deduplicateByLookup(exerciseDataFrame,variations=uniqueExercise,
                                                   targetColumn="Exercise Name",lookupColumns="Equipment")

    # Now, we use multi-encoding on equipment.
    uniqueEquipment = getUniqueValuesInColumn(exerciseDataFrame, columnName="Equipment", isValueList=False)
    exerciseDataFrame = multiHotEncodeAndMergeEquipment(exerciseDataFrame)

    # Now we apply any manual renaming changes as specified in manual_exercise_renames.csv
    exerciseDataFrame = applyManualExerciseRenaming(exerciseDataFrame)

    # Also, rename "Main_muscle" to "Muscle Group" for consistency
    exerciseDataFrame.rename(columns={'Main_muscle': 'Muscle Group'}, inplace=True)

    # Add in an arbitrary "time per set", which is identical for all exercises atm.
    exerciseDataFrame["Time Per Set"] = 1

    #for uniqueMuscle in uniqueMuscles:
    #    print(f"{uniqueMuscle}: {exerciseDataFrame[uniqueMuscle].sum()}")

    generateDuplicateExerciseNameReport(exerciseDataFrame, groupColumns=list(uniqueMuscles) + list(uniqueEquipment))
    generateUniqueValueReport(exerciseDataFrame,blacklistColumns=list(uniqueMuscles) + list(uniqueEquipment))

    #endregion === Exercise Data Frame ===

    #region === Other Data Frames

    muscleDataFrame = pd.DataFrame(uniqueMuscles, columns=["Muscle"])
    equipmentDataFrame = pd.DataFrame(uniqueEquipment, columns=["Equipment"])

    # Add in equipment group/setup time data
    equipmentDataFrame = addEquipmentGroupsAndTimes(equipmentDataFrame)

    # Add in muscle group data
    muscleDataFrame = addMuscleGroups(muscleDataFrame)

    #endregion === Other Data Frames

    return exerciseDataFrame, muscleDataFrame, equipmentDataFrame

#endregion === Data Processing ===




#exercise, muscles, equipment = fullProcessData()
#print(exercise[exercise["Wrist Extensors"] > 0].to_string())
