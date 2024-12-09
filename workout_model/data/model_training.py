from workout_model.data.preparation import fullProcessData
from workout_model.data.synthetic_data import genSyntheticWeek,genRandomUserInput
import json
import tensorflow as tf
from tensorflow.keras import layers

#region === Data Flattening/Expanding ===

# This method accepts a data input, and converts it into a sequence to be understood by the seq2seq model.
def sequenceDataInput(dataInput):
    sequence = []

    # Time per day
    sequence.append("TIME_PER_DAY:")
    for day, time in dataInput["time_per_day"].items():
        sequence.extend([day, str(time), "|"])

    # Equipment per day
    sequence.append("EQUIPMENT_PER_DAY:")
    for day, equipment in dataInput["equipment_per_day"].items():
        equipment_str = ",".join(equipment) if equipment else "NO_EQUIPMENT"
        sequence.extend([day, equipment_str, "|"])

    # Blacklist per day
    sequence.append("BLACKLIST_PER_DAY:")
    for day, blacklist in dataInput["blacklist_per_day"].items():
        blacklist_str = ",".join(blacklist) if blacklist else "NO_BLACKLIST"
        sequence.extend([day, blacklist_str, "|"])

    # Base max threshold
    sequence.extend(["BASE_MAX_THRESHOLD:", str(dataInput["base_max_threshold"]), "|"])

    # Target sets per muscle
    sequence.extend(["TARGET_SETS_PER_MUSCLE:", str(dataInput["target_sets_per_muscle"]), "|"])

    return sequence
# This method accepts a data output, and converts it into a sequence to be understood by the seq2seq model.
def sequenceDataOutput(dataOutput):
    sequence = []

    for day, details in dataOutput.items():
        # Day header
        sequence.extend(["DAY:", day, "|"])

        # Time available
        sequence.extend(["TIME_AVAILABLE:", str(details["time"]), "|"])

        # Equipment available
        equipment_str = ",".join(details["equipmentAvailable"])
        sequence.extend(["EQUIPMENT_AVAILABLE:", equipment_str, "|"])

        # Exercise blacklist
        blacklist_str = ",".join(details["exerciseBlacklist"]) if details["exerciseBlacklist"] else "NO_BLACKLIST"
        sequence.extend(["EXERCISE_BLACKLIST:", blacklist_str, "|"])

        # Maximum muscle usage threshold
        sequence.extend(["MAX_THRESHOLD:", str(details["maximumMuscleUsageThreshold"]), "|"])

        # Routine (simplified)
        sequence.append("ROUTINE:")
        for exercise in details["routine"][0]:  # Only need exercise names and configs
            sequence.extend([exercise["name"], exercise["config"], "|"])

    return sequence

#endregion === Data Flattening/Expanding ===


exercisesDF, musclesDF, equipmentDF = fullProcessData()
for i in range(1):
    userInput = genRandomUserInput(exercisesDF,equipmentDF)
    routine = genSyntheticWeek(exercisesDF,musclesDF,equipmentDF,time_per_day=userInput["time_per_day"],
                               equipment_per_day=userInput["equipment_per_day"],
                               blacklist_per_day=userInput["blacklist_per_day"],
                               base_max_threshold=userInput["base_max_threshold"],
                               target_sets_per_muscle=userInput["target_sets_per_muscle"])
    #print(userInput)
    #display_weekly_schedule(routine,musclesDataFrame=musclesDF)
    #print("\n\n\n\n")
    print(f"{i}\n{userInput}\n{routine}\n\n")
    print("\n\n\n")
    print(sequenceDataInput(userInput))
    print(sequenceDataOutput(routine))
    with open("temp.json","w") as f:
        json.dump({"user_input":userInput,"model_output":routine},f,indent=3)
