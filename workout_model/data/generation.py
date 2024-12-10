import random
import math
import json
from workout_model.data.preparation import fullProcessData
from workout_model.common.paths import paths

exercises, muscles, equipment = fullProcessData()


#region === Data Generation ===

def generateRoutine(exerciseDataFrame,muscleDataFrame,equipmentDataFrame,
    dayDict : dict, exerciseBlacklist : list = None):

    #if(not equipmentAvailable):
    #    equipmentAvailable = ["Body Weight","Isometric","Self-assisted"]
    if(not exerciseBlacklist):
        exerciseBlacklist = []

    with open(paths["data"] / "muscle_importance.json") as f:
        muscleImportance = json.load(f)

    # 1. Determine exactly how many slots are available for the week
    daysAvailable = 0
    numSlots = 0
    for day, time in dayDict.items():
        numSlots += time / 3
        if(time > 0):
            daysAvailable += 1
    allSlots = []
    for slot in range(numSlots):
        allSlots.append({"Exercise1" : None, "Exercise2" : None})


    print(f"numSlots: {numSlots}")
    print(f"daysAvailable: {daysAvailable}")

    # 2. Using total time available and total slots available, come up with an exact usage amount for each muscle for the week, given that we have between slots to slots*2 available sets
    priorityMuscleCount = 0
    priorityMuscles = set()
    for muscle,importance in muscleImportance.items():
        if(importance == "priority"):
            priorityMuscleCount += 1
            priorityMuscles.add(muscle)
    setsPerMuscle = int((numSlots * 2) / priorityMuscleCount)

    print(f"priorityMuscleCount: {priorityMuscleCount}")
    print(f"setsPerMuscle: {setsPerMuscle}")

    filteredExercises = exerciseDataFrame[
        #(exerciseDataFrame[equipmentAvailable].sum(axis=1) > 0) &  # Check valid equipment usage
        ~exerciseDataFrame["Exercise Name"].isin(exerciseBlacklist)  # Blacklisted exercises
        ]
    # Helper method to get a filter of valid exercises for a muscle
    def get_valid_exercises(targetMuscle,_muscleUsage):
        valid_exercises = []
        for _, _exercise in filteredExercises.iterrows():
            if _exercise[targetMuscle] == 1.0:
                valid_exercises.append(_exercise)
        return valid_exercises

    # 3. Iterate through each 3 set grouping for each muscle. Select one exercise, add any unexpected target muscles, continue until all muscles have been hit or time has run out. prioritize compound whenever possible UNTIL time has run out, then switch to prioritizing iso
    muscleUsage = {}
    usedSlots = 0
    adjustedFinalMuscles = priorityMuscles
    for adjustedFinalMuscle in adjustedFinalMuscles:
        muscleUsage[adjustedFinalMuscle] = setsPerMuscle
    while usedSlots <= numSlots:
        nextMuscle = [priorityMuscle for priorityMuscle,usage in muscleUsage.items() if usage == max(muscleUsage.values())][0]
        validExercises = get_valid_exercises(targetMuscle=nextMuscle,_muscleUsage=muscleUsage)
        # In case the muscle has no valid exercises, the user just won't have the muscle worked in this routine.
        if(not validExercises):
            muscleUsage[nextMuscle] = 0.0
            continue
        if(muscleUsage[nextMuscle] == 0):
            break
        setAmount = 3 if muscleUsage[nextMuscle] >= 3 else muscleUsage[nextMuscle]



        allUsedTargetMuscles = set()
        for i in range(100):
            # First, try to find a good slot with one EXISTING exercise to add this exercise to, this is compatible
            for slot in allSlots:
                existingExercise = slot["Exercise1"]

            # TODO finish this
            chosenExercise = random.choice(validExercises)
            # Iterate through all target muscles in the chosenExercise, and add it to our usage dict
            for priorityMuscle in priorityMuscles:
                if chosenExercise[priorityMuscle] == 1.0:
                    allUsedTargetMuscles.add(priorityMuscle)

            for userTargetMuscle in allUsedTargetMuscles:
                if(muscleUsage[userTargetMuscle] < setAmount):
                    continue

        for usedTargetMuscle in allUsedTargetMuscles:
            muscleUsage[usedTargetMuscle] -= setAmount

        print(f"usedSlotsBEFORE: {usedSlots} | ",end="")
        usedSlots += setAmount
        selectedExercises.append({"Exercise":chosenExercise,"Sets":setAmount})
        print(f"{nextMuscle}: usage: {muscleUsage[nextMuscle]} | usedSlotsAFTER: {usedSlots}")

    return selectedExercises, muscleUsage


# 4. Attempt to bind exercise groups (3 sets) to another compatible exercise group wherever possible, forming supersets. All leftovers become regular sets.


# 5. Distribute all exercise groups across the week in whatever combination doesn't violate exercise maximum threshold


userSchedule = {"Monday": 30, "Tuesday": 30, "Wednesday": 45, "Thursday": 0, "Friday": 30, "Saturday": 0, "Sunday": 0}
selExercised, musUsage = generateRoutine(exercises,muscles,equipment,
                dayDict=userSchedule)
print("hello")