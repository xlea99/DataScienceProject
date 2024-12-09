from workout_model.data.preparation import fullProcessData
import random
import math

#region === Synthetic Generators ===

# This method generates a full synthetic day routine, given the listed variables. It will always conform to boundaries,
# but will semi-randomly select exercises to provide rich variety to data.
def genSyntheticDay(exerciseDataFrame,musclesDataFrame,equipmentDataFrame,time,musclesToHit,maximumMuscleUsageThreshold=8.0,
                    equipmentAvailable=None,  # None means all equipment is available
                    exerciseBlacklist=None,  # None means no exercises are blacklisted
                    max_attempts=100):  # Maximum attempts to add exercises to prevent infinite loops
    # Normalize inputs
    if equipmentAvailable is None:
        equipmentAvailable = equipmentDataFrame["Equipment"].tolist()  # All equipment available
    if exerciseBlacklist is None:
        exerciseBlacklist = []
    elif isinstance(exerciseBlacklist, str):
        exerciseBlacklist = [exerciseBlacklist]

    # Define set/rest configurations
    set_rest_configs = {
        "Single Set": {"sets": 1, "rest_time": 2, "total_time": 3},
        "Single Superset": {"sets": 2, "rest_time": 1, "total_time": 3},
        "Double Set": {"sets": 2, "rest_time": 4, "total_time": 6},
        "Double Superset": {"sets": 4, "rest_time": 2, "total_time": 6},
    }

    # Filter exercises based on equipment and blacklist
    filtered_exercises = exerciseDataFrame[
        (exerciseDataFrame[equipmentAvailable].sum(axis=1) > 0) &  # Check valid equipment usage
        ~exerciseDataFrame["Exercise Name"].isin(exerciseBlacklist)  # Blacklisted exercises
    ]

    # Initialize tracking variables
    routine = []
    total_time = 0
    all_muscles = musclesDataFrame["Muscle"].tolist()
    muscle_usage = {muscle: 0 for muscle in all_muscles}

    # Helper: Find least-utilized muscles from the target list
    def get_least_utilized_muscle():
        target_usages = {muscle: muscle_usage[muscle] for muscle in musclesToHit}
        min_usage = min(target_usages.values())
        least_utilized = [muscle for muscle, usage in target_usages.items() if usage == min_usage]
        return random.choice(least_utilized)

    # Helper: Filter valid exercises for a muscle
    def get_valid_exercises(target_muscle):
        valid_exercises = []
        for _, exercise in filtered_exercises.iterrows():
            if exercise[target_muscle] > 0:  # Check if the exercise targets the muscle
                # Ensure no muscle in this exercise exceeds the threshold
                would_exceed_threshold = any(
                    muscle_usage[muscle] + exercise[muscle] * max(config["sets"] for config in set_rest_configs.values()) > maximumMuscleUsageThreshold
                    for muscle in all_muscles if muscle in exercise
                )
                if not would_exceed_threshold:
                    valid_exercises.append(exercise)
        return valid_exercises

    # Main loop: Generate the routine
    attempts = 0
    while total_time < time:
        # Break if too many attempts have been made
        if attempts >= max_attempts:
            print("Max attempts reached. Ending routine generation.")
            break

        # Step 1: Attempt to target the least-utilized muscle
        target_muscle = get_least_utilized_muscle()
        valid_exercises = get_valid_exercises(target_muscle)

        # If no valid exercises exist, try another muscle
        if not valid_exercises:
            musclesToHit = [muscle for muscle in musclesToHit if get_valid_exercises(muscle)]
            if not musclesToHit:
                break  # Exit if no valid exercises exist for any target muscle
            attempts += 1
            continue

        # Randomly select an exercise
        exercise = random.choice(valid_exercises)

        # Prioritize configurations
        valid_configs = [
            (config_name, config)
            for config_name, config in set_rest_configs.items()
            if total_time + config["total_time"] <= time
        ]

        # If no valid configurations exist, retry
        if not valid_configs:
            attempts += 1
            continue

        # Randomly select a configuration from the valid options
        config_name, config = random.choice(valid_configs)

        # Add the exercise to the routine
        routine.append({
            "name": exercise["Exercise Name"],
            "sets": config["sets"],
            "rest_time": config["rest_time"],
            "total_time": config["total_time"],
            "config": config_name,
            "muscles_hit": {muscle: exercise[muscle] * config["sets"] for muscle in all_muscles if exercise[muscle] > 0},
            "equipment_used": [equipment for equipment in equipmentAvailable if exercise.get(equipment, 0) > 0]
        })
        total_time += config["total_time"]

        # Update muscle usage
        for muscle in all_muscles:
            if muscle in exercise:
                muscle_usage[muscle] += exercise[muscle] * config["sets"]

    return routine, muscle_usage

# This method generates a full synthetic week map, and then generates a day routine for each valid weekday, given
# a user schedule.
def genSyntheticWeek(exerciseDataFrame,musclesDataFrame,equipmentDataFrame,
        time_per_day,  # Dictionary: {"Monday": 45, "Tuesday": 30, ...}
        equipment_per_day,  # Dictionary: {"Monday": ["Dumbbell", ...], ...}
        blacklist_per_day,  # Dictionary: {"Monday": ["Push-up", ...], ...}
        base_max_threshold=8.0,  # Default daily threshold
        target_sets_per_muscle=18):  # Target weekly sets per muscle
    # Days of the week
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Extract all unique muscles
    all_muscles = musclesDataFrame["Muscle"].tolist()

    # Filter for days with actual time available
    valid_days = [day for day in days if time_per_day.get(day, 0) > 0]
    total_days_available = len(valid_days)
    total_time_available = sum(time_per_day.get(day, 0) for day in valid_days)

    # Estimate required time to hit all muscles once
    estimated_required_time = len(all_muscles) * (target_sets_per_muscle / 7) * 3  # Average 3 mins per set/rest config

    # Determine surplus ratio
    time_surplus_ratio = total_time_available / estimated_required_time

    # Distribute muscles across available days
    muscles_per_day = math.ceil(len(all_muscles) / total_days_available)
    muscle_assignments = {day: [] for day in valid_days}

    shuffled_muscles = all_muscles.copy()
    random.shuffle(shuffled_muscles)

    # Initial assignment
    for i, muscle in enumerate(shuffled_muscles):
        muscle_assignments[valid_days[i % total_days_available]].append(muscle)

    # Reassign some muscles if surplus time allows
    if time_surplus_ratio > 1.3:  # Allow reassigning if surplus time is significant
        surplus_days = [
            day for day in valid_days if time_per_day[day] > (estimated_required_time / total_days_available)
        ]

        extra_muscles = shuffled_muscles.copy()
        random.shuffle(extra_muscles)

        for day in surplus_days:
            muscles_to_reassign = extra_muscles[:muscles_per_day // 2]  # Add ~50% of initial daily muscles
            muscle_assignments[day].extend(muscles_to_reassign)
            extra_muscles = extra_muscles[muscles_per_day // 2:]

    # Calculate maximumMuscleUsageThreshold for each day
    def calculate_max_threshold(days_available, time_available):
        if days_available >= 6:
            return base_max_threshold  # Conservative for many days
        elif 4 <= days_available < 6:
            return base_max_threshold + 2  # Moderate flexibility
        elif 2 <= days_available < 4:
            return base_max_threshold + 4  # High flexibility
        else:  # 1 day available
            return base_max_threshold + 6  # Very high flexibility

    # Prepare arguments for each day
    week_schedule = {}
    for day in valid_days:
        time_available = time_per_day[day]
        equipment_available = equipment_per_day.get(day, [])
        exercise_blacklist = blacklist_per_day.get(day, [])

        # Calculate the max threshold based on the total days and available time
        max_threshold = calculate_max_threshold(total_days_available, time_available)

        # Assign muscles to this day
        muscles_to_hit = muscle_assignments[day]

        # Create the day's argument list
        week_schedule[day] = {
            "time": time_available,
            "equipmentAvailable": equipment_available,
            "exerciseBlacklist": exercise_blacklist,
            "musclesToHit": muscles_to_hit,
            "maximumMuscleUsageThreshold": max_threshold,
        }

    # Generate an actual day routine for each day.
    for day in valid_days:
        week_schedule[day]["routine"] = genSyntheticDay(exerciseDataFrame,musclesDataFrame,equipmentDataFrame,
                                                   time=week_schedule[day]["time"],
                                                   musclesToHit=week_schedule[day]["musclesToHit"],
                                                   maximumMuscleUsageThreshold=week_schedule[day]["maximumMuscleUsageThreshold"],
                                                   equipmentAvailable=week_schedule[day]["equipmentAvailable"],
                                                   exerciseBlacklist=week_schedule[day]["exerciseBlacklist"])

    return week_schedule

# This method generates a random single user's input variables for use with the genSyntheticWeek generator.
def genRandomUserInput(exerciseDataFrame, equipmentDataFrame):
    # 1. Days available
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    num_days_available = random.randint(1, 7)  # Randomly pick number of workout days
    active_days = random.sample(days_of_week, num_days_available)  # Randomly select active days

    # 2. User attributes: time volatility and baseline time availability
    time_volatility = random.randint(0, 5)  # 0 = very consistent, 5 = highly variable
    baseline_time_free = random.randint(1, 5)  # 1 = low time available, 5 = ample time available

    # Generate time_per_day based on volatility and baseline
    time_per_day = {}
    for day in days_of_week:
        if day in active_days:
            base_time = baseline_time_free * 15  # Base time in minutes (e.g., 15-75 mins)
            if time_volatility == 0:
                time_available = base_time
            else:
                time_available = max(15, base_time + random.randint(-10 * time_volatility, 10 * time_volatility))
            time_per_day[day] = time_available
        else:
            time_per_day[day] = 0  # No workout on this day

    # 3. Equipment availability per day
    no_equipment = ["Body Weight", "Isometric", "Self-assisted"]
    isNoEquipmentUser = random.randrange(1,5) == 1
    isUnrelatedEquipmentUser = random.randrange(1, 6) == 1
    isConsistentEquipmentUser = random.randrange(1,4) != 1
    consistentEquipment = set()
    consistentEquipmentHasBeenGenerated = False
    equipment_groups = equipmentDataFrame["Equipment"].unique().tolist()
    equipment_per_day = {}

    for day in days_of_week:
        if day in active_days:
            # Always include no-equipment group
            day_equipment = set(no_equipment)

            if(not isNoEquipmentUser):
                # Add 1-4 equipment groups
                num_groups = random.randint(1, 4)
                selected_groups = random.sample(equipment_groups, num_groups)
                if(isConsistentEquipmentUser):
                    if(consistentEquipmentHasBeenGenerated):
                        day_equipment.union(consistentEquipment)
                    else:
                        consistentEquipment.update(selected_groups)
                        day_equipment.union(consistentEquipment)
                else:
                    day_equipment.update(selected_groups)

            if (isUnrelatedEquipmentUser):
                # Add 1-5 random standalone equipment for variety
                standalone_equipment = random.sample(equipment_groups, random.randint(1, 5))
                if(isConsistentEquipmentUser):
                    consistentEquipment.update(standalone_equipment)
                    day_equipment.union(consistentEquipment)
                else:
                    day_equipment.union(standalone_equipment)

            equipment_per_day[day] = list(day_equipment)
        else:
            equipment_per_day[day] = []

    # 4. Exercise blacklists per day
    blacklist_neediness = random.randint(0, 3)  # 0 = no blacklist, 3 = very needy
    blacklist_per_day = {}

    if blacklist_neediness == 0:
        # No blacklist
        blacklist_per_day = {day: [] for day in days_of_week}
    else:
        # Generate blacklists
        all_exercises = exerciseDataFrame["Exercise Name"].unique().tolist()
        uniform_blacklist = random.random() > 0.5  # 50% chance for uniform blacklist

        if uniform_blacklist:
            common_blacklist = random.sample(all_exercises, random.randint(5, 15))
            for day in days_of_week:
                blacklist_per_day[day] = common_blacklist if day in active_days else []
        else:
            for day in days_of_week:
                if day in active_days:
                    blacklist_per_day[day] = random.sample(all_exercises, random.randint(5, 15))
                else:
                    blacklist_per_day[day] = []

    # 5. Base thresholds
    base_max_threshold = random.randrange(5,14) if random.randrange(1,6) == 1 else 8.0
    target_sets_per_muscle = random.randrange(8,24) if random.randrange(1,6) == 1 else 18.0

    return {
        "time_per_day": time_per_day,
        "equipment_per_day": equipment_per_day,
        "blacklist_per_day": blacklist_per_day,
        "base_max_threshold": base_max_threshold,
        "target_sets_per_muscle": target_sets_per_muscle,
    }

#endregion === Synthetic Generators ===
#region === Displays ===

# Helper functions for neatly displaying the results of daily/weekly routines.
def display_routine(routine, exerciseDataFrame, muscleDataFrame, show_muscle_performance=False):
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

    # Initialize a dictionary to track total muscle usage
    muscle_performance = {muscle: 0 for muscle in muscle_columns}

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

        # Update muscle performance totals
        for muscle, value in muscles.items():
            muscle_performance[muscle] += value * exercise["sets"]

        # Display exercise information
        print(f"{i}. {exercise['name']}")
        print(f"   Configuration: {exercise['config']}")
        print(f"   Sets: {exercise['sets']} | Rest Time: {exercise['rest_time']} min | Total Time: {exercise['total_time']} min")
        print("   Muscles Hit:")
        for muscle, value, group in muscles_with_groups:
            print(f"      - {muscle} ({group}): {value * exercise['sets']:.1f}")
        print(f"   Valid Equipment: {', '.join(equipment) if equipment else 'None'}")
        print("-" * 50)

    print("\nSummary:")
    print(f"Total Exercises: {len(routine)}")
    print(f"Total Time: {sum(ex['total_time'] for ex in routine):.1f} minutes")
    print("=" * 50)

    # Muscle Performance Review (optional)
    if show_muscle_performance:
        print("\nMuscle Performance Review")
        print("=" * 50)

        # Filter and sort muscles by total usage (descending), excluding unused muscles
        used_muscle_performance = [
            (muscle, usage) for muscle, usage in muscle_performance.items() if usage > 0
        ]
        sorted_muscle_performance = sorted(
            used_muscle_performance, key=lambda x: -x[1]
        )

        for muscle, usage in sorted_muscle_performance:
            muscle_group = muscleDataFrame[muscleDataFrame["Muscle"] == muscle]["Muscle Group"].iloc[0]
            print(f"{muscle} ({muscle_group}): {usage:.1f} sets")

        print("=" * 50)
def display_weekly_schedule(week_schedule, musclesDataFrame):
    print("=" * 50)
    print("WEEKLY WORKOUT SCHEDULE")
    print("=" * 50)

    # Initialize muscle hit tracking
    muscle_hit_summary = {muscle: 0 for muscle in musclesDataFrame["Muscle"].tolist()}

    # Loop through each day and display the schedule
    for day, schedule in week_schedule.items():
        print(f"{day.upper()}")
        print("-" * 50)
        print(f"Time Available: {schedule['time']} minutes")
        print(f"Equipment Available: {', '.join(schedule['equipmentAvailable']) if schedule['equipmentAvailable'] else 'None'}")
        print(f"Exercise Blacklist: {', '.join(schedule['exerciseBlacklist']) if schedule['exerciseBlacklist'] else 'None'}")
        print(f"Maximum Muscle Usage Threshold: {schedule['maximumMuscleUsageThreshold']}")
        print("Muscles to Hit:")
        muscles_with_groups = [
            (muscle, musclesDataFrame[musclesDataFrame["Muscle"] == muscle]["Muscle Group"].iloc[0])
            for muscle in schedule["musclesToHit"]
        ]
        for muscle, group in muscles_with_groups:
            print(f"   - {muscle} ({group})")

        # Update muscle hit summary
        for muscle in schedule["musclesToHit"]:
            muscle_hit_summary[muscle] += 1

        print("=" * 50)

    # Display muscle hit summary
    print("\nMUSCLE PERFORMANCE REVIEW")
    print("=" * 50)
    print(f"{'Muscle':<30}{'Muscle Group':<20}{'Days Hit':>10}")
    print("-" * 60)
    for muscle in musclesDataFrame["Muscle"].tolist():
        days_hit = muscle_hit_summary[muscle]
        muscle_group = musclesDataFrame[musclesDataFrame["Muscle"] == muscle]["Muscle Group"].iloc[0]
        print(f"{muscle:<30}{muscle_group:<20}{days_hit:>10}")
    print("=" * 50)

#endregion === Displays ===

exercisesDF, musclesDF, equipmentDF = fullProcessData()


for i in range(50):
    userInput = genRandomUserInput(exercisesDF,equipmentDF)
    routine = genSyntheticWeek(exercisesDF,musclesDF,equipmentDF,time_per_day=userInput["time_per_day"],
                               equipment_per_day=userInput["equipment_per_day"],
                               blacklist_per_day=userInput["blacklist_per_day"],
                               base_max_threshold=userInput["base_max_threshold"],
                               target_sets_per_muscle=userInput["target_sets_per_muscle"])
    print(userInput)
    display_weekly_schedule(routine,musclesDataFrame=musclesDF)
    print("\n\n\n\n")
