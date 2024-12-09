from workout_model.data.preparation import fullProcessData
import random

# Day-level routine generator
def generate_day_routine(exerciseDataFrame, time, equipmentAvailable,
                         targetMuscleUsageThreshold=3.0, maximumMuscleUsageThreshold=8.0,
                         compoundSetCount=3, isoSetCount=2,
                         exerciseBlacklist: (str, list) = None, musclesToHit: (str, list) = None):
    # Normalize inputs
    if exerciseBlacklist is None:
        exerciseBlacklist = []
    elif isinstance(exerciseBlacklist, str):
        exerciseBlacklist = [exerciseBlacklist]

    if musclesToHit is None:
        musclesToHit = []
    elif isinstance(musclesToHit, str):
        musclesToHit = [musclesToHit]

    # Step 1: Filter exercises by equipment availability, blacklist, and target muscles
    filtered_exercises = exerciseDataFrame[
        (exerciseDataFrame[[eq for eq in equipmentAvailable]].sum(axis=1) > 0) &
        ~exerciseDataFrame["Exercise Name"].isin(exerciseBlacklist)
    ]
    if musclesToHit:
        filtered_exercises = filtered_exercises[
            filtered_exercises[musclesToHit].sum(axis=1) > 0
        ]

    # Initialize routine and tracking variables
    routine = []
    total_time = 0
    muscle_usage = {muscle: 0 for muscle in exerciseDataFrame.columns if muscle not in equipmentAvailable and exerciseDataFrame[muscle].dtype in [int, float]}

    def can_add_exercise(exercise, set_count):
        """Check if adding this exercise would exceed maximum usage for any muscle."""
        for muscle, usage in muscle_usage.items():
            projected_usage = usage + exercise.get(muscle, 0) * set_count
            if projected_usage > maximumMuscleUsageThreshold:
                return False
        return True

    def prioritize_exercises(exercises, set_count):
        """Prioritize exercises based on how close they bring muscles to target."""
        valid_exercises = []
        for _, exercise in exercises.iterrows():
            if can_add_exercise(exercise, set_count):
                # Calculate how much closer this exercise brings muscles to target
                benefit = sum(
                    max(0, targetMuscleUsageThreshold - (muscle_usage[muscle] + exercise.get(muscle, 0) * set_count))
                    for muscle in muscle_usage.keys()
                )
                valid_exercises.append((exercise, benefit))
        # Sort by descending benefit
        valid_exercises.sort(key=lambda x: -x[1])
        return valid_exercises

    # Step 2: Prioritize compound exercises
    compound_exercises = filtered_exercises[filtered_exercises["Mechanics"] == "Compound"]
    while total_time < time:
        prioritized_compounds = prioritize_exercises(compound_exercises, compoundSetCount)
        if not prioritized_compounds:
            break  # No more valid exercises
        exercise, _ = random.choice(prioritized_compounds[:3])  # Pick one of the top 3 randomly
        routine.append({"name": exercise["Exercise Name"], "sets": compoundSetCount, "time": exercise["Time Per Set"] * compoundSetCount})
        total_time += exercise["Time Per Set"] * compoundSetCount
        for muscle in muscle_usage.keys():
            muscle_usage[muscle] += exercise.get(muscle, 0) * compoundSetCount

    # Step 3: Fill remaining time with isolation exercises
    isolation_exercises = filtered_exercises[filtered_exercises["Mechanics"] == "Isolated"]
    while total_time < time:
        prioritized_isolations = prioritize_exercises(isolation_exercises, isoSetCount)
        if not prioritized_isolations:
            break  # No more valid exercises
        exercise, _ = random.choice(prioritized_isolations[:3])  # Pick one of the top 3 randomly
        routine.append({"name": exercise["Exercise Name"], "sets": isoSetCount, "time": exercise["Time Per Set"] * isoSetCount})
        total_time += exercise["Time Per Set"] * isoSetCount
        for muscle in muscle_usage.keys():
            muscle_usage[muscle] += exercise.get(muscle, 0) * isoSetCount

    return routine, muscle_usage

# Helper function for neatly displaying the results of a routine.
def display_routine(routine, exerciseDataFrame, muscleDataFrame):
    print("=" * 50)
    print("WORKOUT ROUTINE")
    print("=" * 50)

    # Identify muscle columns from the muscleDataFrame
    muscle_columns = muscleDataFrame["Muscle"].tolist()

    # Identify equipment columns as all other binary columns in the exerciseDataFrame
    equipment_columns = [
        col for col in exerciseDataFrame.columns
        if col not in muscle_columns and exerciseDataFrame[col].dtype in [int, float] and exerciseDataFrame[col].max() == 1
    ]

    for i, exercise in enumerate(routine, start=1):
        # Get additional details from the dataframe
        exercise_details = exerciseDataFrame[exerciseDataFrame["Exercise Name"] == exercise["name"]].iloc[0]

        # Extract muscles and their engagement levels
        muscles = {
            col: exercise_details[col]
            for col in muscle_columns if col in exerciseDataFrame.columns and exercise_details[col] > 0
        }

        # Sort muscles by engagement level (1.0 first, then 0.5)
        sorted_muscles = sorted(muscles.items(), key=lambda x: -x[1])

        # Add muscle group information
        muscles_with_groups = [
            (muscle, value, muscleDataFrame[muscleDataFrame["Muscle"] == muscle]["Muscle Group"].iloc[0])
            for muscle, value in sorted_muscles
        ]

        # Extract equipment used (filter only equipment columns)
        equipment = [col for col in equipment_columns if exercise_details.get(col, 0) == 1]

        # Display exercise information
        print(f"{i}. {exercise['name']}")
        print(f"   Sets: {exercise['sets']} | Time: {exercise['time']} min")
        print("   Muscles Hit:")
        for muscle, value, group in muscles_with_groups:
            print(f"      - {muscle} ({group}): {value}")
        print(f"   Equipment Used: {', '.join(equipment) if equipment else 'None'}")
        print("-" * 50)

    print("\nSummary:")
    print(f"Total Exercises: {len(routine)}")
    print(f"Total Time: {sum(ex['time'] for ex in routine):.1f} minutes")
    print("=" * 50)


# Build dataframes
exercisesDF, musclesDF, equipmentDF = fullProcessData()

weekly_muscles_hit = {
    "Gastrocnemius": 0, "Latissimus Dorsi": 0, "Teres Minor": 0, "Gluteus Maximus": 0,
    "Hip Abductors": 0, "Supraspinatus": 0, "Upper Trapezius": 0, "Splenius": 0,
    "Hip Flexors": 0, "Wrist Flexors": 0, "Infraspinatus": 0, "Teres Major": 0,
    "Quadriceps": 0, "Levator Scapulae": 0, "Triceps Brachii": 0, "Anterior Deltoid": 0,
    "Lateral Deltoid": 0, "Pectoralis Major Sternal": 0, "Pectoralis Major Clavicular": 0,
    "Rhomboids": 0, "Lower Trapezius": 0, "Brachialis": 0, "Soleus": 0,
    "Serratus Anterior": 0, "Posterior Deltoid": 0, "Tibialis Anterior": 0, "Hip Adductors": 0,
    "Wrist Extensors": 0, "Biceps Brachii": 0, "Sternocleidomastoid": 0, "Supinator": 0,
    "Brachioradialis": 0, "Subscapularis": 0, "Middle Trapezius": 0, "Hamstrings": 0, "Pronators": 0,
}
availableEquipment = ["Dumbbell", "Body Weight"]
blacklistedExercises = []
musclesToTarget = ["Latissimus Dorsi",
        "Posterior Deltoid",
        "Rhomboids",
        "Upper Trapezius",
        "Middle Trapezius",
        "Lower Trapezius",
        "Teres Major",
        "Teres Minor",
        "Infraspinatus",
        "Subscapularis",
        "Biceps Brachii",
        "Triceps Brachii",
        "Brachialis"]

# Generate day routine
thisRoutine, updated_muscles_hit = generate_day_routine(exercisesDF,time=12,equipmentAvailable=availableEquipment,
                                                        exerciseBlacklist=blacklistedExercises,musclesToHit=musclesToTarget)

display_routine(thisRoutine,exercisesDF,musclesDF)
print("\n\n\n")
print(updated_muscles_hit)