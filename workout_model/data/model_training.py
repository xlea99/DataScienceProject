from workout_model.data.preparation import fullProcessData
from workout_model.common.paths import paths
from workout_model.data.synthetic_data import genSyntheticWeek,genRandomUserInput,display_weekly_schedule
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import os


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
#region === Efficiency Scoring ===

# Calculates how close to the target muscle usage the total workout was, for each muscle, with a heavy bias
# towards priority muscles.
def genTargetMuscleSetAdherenceScore(musclesDataFrame, userInputDict, routineOutputDict):
    # Load muscle importance data
    muscle_importance_path = paths["data"] / "muscle_importance.json"
    with open(muscle_importance_path, 'r') as f:
        muscle_importance = json.load(f)

    target_sets = userInputDict["target_sets_per_muscle"]
    muscle_performance = {day: data["routine"][1] for day, data in routineOutputDict.items()}

    # Aggregate actual sets across all days
    total_sets_per_muscle = {}
    for daily_performance in muscle_performance.values():
        for muscle, sets in daily_performance.items():
            total_sets_per_muscle[muscle] = total_sets_per_muscle.get(muscle, 0) + sets

    # Calculate penalties for target sets
    total_penalty = 0
    muscle_weights = {"priority": 5, "accessory": 1}

    for muscle, actual_sets in total_sets_per_muscle.items():
        importance = muscle_importance.get(muscle, "accessory")
        weight = muscle_weights[importance]
        penalty = weight * abs(actual_sets - target_sets)
        total_penalty += penalty

    # Normalize the penalty score by the number of muscles
    num_muscles = len(musclesDataFrame["Muscle"].unique())
    normalized_score = 1 - (total_penalty / (num_muscles * target_sets * muscle_weights["priority"]))

    # Return normalized efficiency score
    return max(0, normalized_score)  # Ensure score is not negative
# Calculates muscle uniformity index, which measures how uniform the work on all target muscles is, penalizing higher
# variance.
def genMuscleUniformityScore(routineOutputDict):
    # Load muscle importance
    with open(paths["data"] / "muscle_importance.json", "r") as f:
        muscle_importance = json.load(f)

    # Filter for priority muscles
    priority_muscles = [
        muscle for muscle, importance in muscle_importance.items() if importance == "priority"
    ]

    # Extract the musclePerformance from routineOutputDict
    muscle_performance = {}
    for day_data in routineOutputDict.values():
        daily_performance = day_data.get("routine", ([], {}))[1]
        for muscle, sets in daily_performance.items():
            muscle_performance[muscle] = muscle_performance.get(muscle, 0) + sets

    # Extract sets for priority muscles
    priority_sets = [
        muscle_performance.get(muscle, 0) for muscle in priority_muscles
    ]

    # If no priority muscles were worked, return 0
    if not any(priority_sets):
        return 0.0

    # Calculate mean and variance
    mean_sets = sum(priority_sets) / len(priority_sets)
    variance = sum((sets - mean_sets) ** 2 for sets in priority_sets) / len(priority_sets)

    # Transform variance into a score
    uniformity_score = 1 / (1 + variance)

    return uniformity_score
# Calculates the Stretch Potential metric for the weekly routine.
def genStretchPotential(routineOutputDict):
    # Load exercise configurations
    with open(paths["data"] / "exercise_configurations.json", "r") as f:
        exercise_configurations = {ex["name"]: ex["stretch_factor"] for ex in json.load(f)["groups"]}

    # Initialize variables
    total_days = 0
    total_stretch_score = 0.0

    # Process each day in the routine
    for day_name, day_data in routineOutputDict.items():
        day_routine = day_data.get("routine", ([], {}))[0]  # Get list of exercises
        if not day_routine:
            continue  # Skip days with no exercises

        # Calculate average stretch factor for the day
        day_stretch_factors = [
            exercise_configurations.get(exercise["name"], 0.0) for exercise in day_routine
        ]
        day_average_stretch = sum(day_stretch_factors) / len(day_stretch_factors)

        # Update overall metrics
        total_days += 1
        total_stretch_score += day_average_stretch

    # Return average stretch factor across all days
    if total_days == 0:
        return 0.0  # No valid days in the routine

    return total_stretch_score / total_days

# Combine various metrics to generate a single efficiency score for an input/output.
def genEfficiencyScore(musclesDataFrame,userInputDict,routineOutputDict,
                       uniformityWeight=0.1,setAdherenceWeight=0.1,stretchPotentialWeight=0.1):
    # Generate individual scores
    muscleUniformityScore = genMuscleUniformityScore(routineOutputDict)
    targetMuscleSetAdherenceScore = genTargetMuscleSetAdherenceScore(musclesDataFrame,userInputDict,routineOutputDict)
    stretchPotentialScore = genStretchPotential(routineOutputDict)

    # Normalize scores to ensure valid ranges (0-1)
    muscleUniformityScore = max(0.0, min(muscleUniformityScore, 1.0))
    targetMuscleSetAdherenceScore = max(0.0, min(targetMuscleSetAdherenceScore, 1))
    stretchPotentialScore = max(0.0, min(stretchPotentialScore, 1.0))

    # Weighted combination of metrics
    combinedScore = (
        uniformityWeight * muscleUniformityScore +
        setAdherenceWeight * targetMuscleSetAdherenceScore +
        stretchPotentialWeight * stretchPotentialScore
        + 0.7
    )

    return combinedScore

#endregion === Efficiency Scoring ===
#region === Training Data ===

# This method simplifies the whole process and generates a dataframe of numSamples synthetic data, formatted
# for model training.
def generateTrainingDataFrame(exercisesDataFrame, musclesDataFrame, equipmentDataFrame, numSamples):
    data = []
    for i in range(numSamples):
        userInput = genRandomUserInput(exercisesDataFrame, equipmentDataFrame)
        routineOutput = genSyntheticWeek(exercisesDataFrame, musclesDataFrame, equipmentDataFrame, time_per_day=userInput["time_per_day"],
                                   equipment_per_day=userInput["equipment_per_day"],
                                   blacklist_per_day=userInput["blacklist_per_day"],
                                   base_max_threshold=userInput["base_max_threshold"],
                                   target_sets_per_muscle=userInput["target_sets_per_muscle"])
        efficiencyScore = genEfficiencyScore(musclesDataFrame, userInput, routineOutput)
        data.append({"input": sequenceDataInput(userInput),
                     "output": sequenceDataOutput(routineOutput),
                     "score": efficiencyScore})
        if(i % 20 == 0):
            print(f"{i}/{numSamples} training data tuples generated...")

    # Save the dataset as a dataframe, and return it.
    df = pd.DataFrame(data)
    return df

#endregion === Training Data ===

def process_data():
    print("Processing data...")
    exercisesDF, musclesDF, equipmentDF = fullProcessData()
    print("Data processing complete.")
    return exercisesDF, musclesDF, equipmentDF

def generate_synthetic_dataset(exercisesDF, musclesDF, equipmentDF, num_samples=50, save_path=None):
    print(f"Generating synthetic dataset with {num_samples} samples...")
    syntheticDF = generateTrainingDataFrame(exercisesDF, musclesDF, equipmentDF, num_samples)
    print(f"Synthetic dataset generated with {len(syntheticDF)} samples.")

    if save_path:
        syntheticDF.to_csv(save_path, index=False)
        print(f"Synthetic dataset saved at {save_path}.")
    return syntheticDF
def load_synthetic_dataset(load_path):
    if os.path.exists(load_path):
        print(f"Loading synthetic dataset from {load_path}...")
        syntheticDF = pd.read_csv(load_path)
        print(f"Synthetic dataset loaded with {len(syntheticDF)} samples.")
        return syntheticDF
    else:
        raise FileNotFoundError(f"Synthetic dataset file not found at {load_path}.")

def prepare_data(syntheticDF):
    print("Splitting data into training and validation sets...")
    train_df, val_df = train_test_split(syntheticDF, test_size=0.2, random_state=42)
    print(f"Training set: {len(train_df)}, Validation set: {len(val_df)}")

    # Add <start> and <end> tokens
    #print("Adding <start> and <end> tokens to output sequences...")
    #train_df["output"] = train_df["output"].apply(lambda x: f"<start> {x} <end>")
    #val_df["output"] = val_df["output"].apply(lambda x: f"<start> {x} <end>")
    #print("Tokens added.")

    return train_df, val_df


def tokenize_and_pad(train_df, val_df):
    """
    Tokenizes and pads input/output sequences for training and validation datasets.
    Adds <start> and <end> tokens to output sequences.

    Args:
        train_df (pd.DataFrame): Training dataframe with "input" and "output" columns.
        val_df (pd.DataFrame): Validation dataframe with "input" and "output" columns.

    Returns:
        tuple: Padded input/output sequences for training and validation,
               along with input and output tokenizers.
    """
    print("Adding <start> and <end> tokens to output sequences...")

    # Add <start> and <end> tokens to output sequences
    train_df["output"] = train_df["output"].apply(lambda x: f"<start> {x} <end>")
    val_df["output"] = val_df["output"].apply(lambda x: f"<start> {x} <end>")
    print("Tokens added.")

    print("Tokenizing and padding sequences...")

    # Initialize tokenizers with oov_token for robustness
    input_tokenizer = Tokenizer(oov_token="<unk>")
    output_tokenizer = Tokenizer(oov_token="<unk>")

    # Fit tokenizers on training data
    input_tokenizer.fit_on_texts(train_df["input"])
    output_tokenizer.fit_on_texts(train_df["output"])

    # Manually enforce <start> and <end> in the output tokenizer vocabulary
    if "<start>" not in output_tokenizer.word_index:
        output_tokenizer.word_index["<start>"] = len(output_tokenizer.word_index) + 1
    if "<end>" not in output_tokenizer.word_index:
        output_tokenizer.word_index["<end>"] = len(output_tokenizer.word_index) + 1

    # Debugging tokenizers
    print("Input Tokenizer Vocabulary Size:", len(input_tokenizer.word_index))
    print("Output Tokenizer Vocabulary Size:", len(output_tokenizer.word_index))
    print("<start> token index:", output_tokenizer.word_index.get("<start>", "Not Found"))
    print("<end> token index:", output_tokenizer.word_index.get("<end>", "Not Found"))

    # Convert training sequences to numerical format
    input_sequences = input_tokenizer.texts_to_sequences(train_df["input"])
    output_sequences = output_tokenizer.texts_to_sequences(train_df["output"])

    # Pad training sequences
    input_padded = pad_sequences(input_sequences, padding="post")
    output_padded = pad_sequences(output_sequences, padding="post")

    # Convert validation sequences to numerical format
    val_input_sequences = input_tokenizer.texts_to_sequences(val_df["input"])
    val_output_sequences = output_tokenizer.texts_to_sequences(val_df["output"])

    # Pad validation sequences
    val_input_padded = pad_sequences(val_input_sequences, padding="post")
    val_output_padded = pad_sequences(val_output_sequences, padding="post")

    # Debugging padded sequences
    print("Example Padded Input Sequence:", input_padded[0])
    print("Example Padded Output Sequence:", output_padded[0])

    print("Tokenization and padding complete.")
    return (input_padded, output_padded, val_input_padded, val_output_padded, input_tokenizer, output_tokenizer)

def build_seq2seq_model(input_vocab_size, output_vocab_size, embedding_dim=256, lstm_units=512):
    print("Building Seq2Seq model...")

    # Encoder
    encoder_inputs = tf.keras.Input(shape=(None,))
    encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, embedding_dim)(encoder_inputs)
    encoder_outputs, state_h, state_c = tf.keras.layers.LSTM(
        lstm_units, return_state=True, return_sequences=False
    )(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = tf.keras.Input(shape=(None,))
    decoder_embedding = tf.keras.layers.Embedding(output_vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM(
        lstm_units, return_sequences=True, return_state=True
    )
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = tf.keras.layers.Dense(output_vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # Seq2Seq Model
    seq2seq_model = tf.keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    print("Seq2Seq model built.")
    return seq2seq_model, encoder_states

def compile_and_train_model(seq2seq_model, input_padded, output_padded, val_input_padded, val_output_padded, train_df, save_path=None):
    print("Compiling model...")

    def weighted_loss_function(y_true, y_pred):
        # Retrieve efficiency scores as weights
        weights = tf.convert_to_tensor(train_df["score"].values, dtype=tf.float32)  # Shape: [total_samples]

        # Adjust weights to match the batch size
        current_batch_size = tf.shape(y_pred)[0]
        sequence_length = tf.shape(y_pred)[1]

        # Reshape weights to match batch size and sequence length
        weights = tf.reshape(weights[:current_batch_size], (current_batch_size, 1))  # Shape: [batch_size, 1]
        weights = tf.tile(weights, [1, sequence_length])  # Shape: [batch_size, sequence_length]

        # Compute base loss (per timestep in sequence)
        base_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        # Apply weights
        weighted_loss = base_loss * weights
        return weighted_loss

    seq2seq_model.compile(optimizer="adam", loss=weighted_loss_function)

    callbacks = []
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        checkpoint_path = os.path.join(save_path, "seq2seq_checkpoint.keras")
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor="val_loss",
                mode="min",
                verbose=1
            )
        )

    print("Training model...")
    history = seq2seq_model.fit(
        [input_padded, output_padded[:, :-1]],
        output_padded[:, 1:],
        batch_size=64,
        epochs=20,
        validation_data=([val_input_padded, val_output_padded[:, :-1]], val_output_padded[:, 1:]),
        callbacks=callbacks
    )

    if save_path:
        model_path = os.path.join(save_path, "seq2seq_model.h5")
        seq2seq_model.save(model_path)
        print(f"Model saved at {model_path}.")
    print("Model training complete.")
    return history
def load_model(load_path):
    from tensorflow.keras.models import load_model

    model_path = os.path.join(load_path, "seq2seq_model.h5")
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model = load_model(model_path, compile=False)
        print("Model loaded successfully.")
        return model
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}.")

def infer(input_seq, encoder_model, decoder_model, input_tokenizer, output_tokenizer, input_padded_length, max_len=100):
    print("Performing inference...")
    # Encode input sequence
    input_seq = input_tokenizer.texts_to_sequences([input_seq])
    input_seq = pad_sequences(input_seq, maxlen=input_padded_length, padding="post")
    states_value = encoder_model.predict(input_seq)

    # Initialize decoder input with <start> token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = output_tokenizer.word_index.get("<start>")

    decoded_sentence = []
    while True:
        # Predict next token
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Fix for 2D output tokens
        sampled_token_index = np.argmax(output_tokens[0, :])  # Extract directly from 2D output
        sampled_token = output_tokenizer.index_word.get(sampled_token_index, "")

        if sampled_token == "<end>" or len(decoded_sentence) >= max_len:
            break

        decoded_sentence.append(sampled_token)

        # Prepare next decoder input
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update decoder states
        states_value = [h, c]

        print("PREDICTION STEP")
        print("Output tokens shape:", output_tokens.shape)
        print("Sampled token index:", sampled_token_index)
        print("Sampled token:", sampled_token)

    return " ".join(decoded_sentence)

def run_workflow():
    # Define paths
    data_save_path = os.path.join(paths["data"], "synthetic_data.csv")
    model_save_path = os.path.join(paths["data"], "model")

    # Step 1: Data processing
    print("Processing data...")
    exercisesDF, musclesDF, equipmentDF = process_data()
    print("Data processing complete.")

    # Step 2: Generate synthetic dataset
    if os.path.exists(data_save_path):
        syntheticDF = load_synthetic_dataset(data_save_path)
    else:
        syntheticDF = generate_synthetic_dataset(
            exercisesDF, musclesDF, equipmentDF, num_samples=30000, save_path=data_save_path
        )

    # Step 3: Prepare training and validation data
    print("Preparing training and validation data...")
    train_df, val_df = prepare_data(syntheticDF)
    print(f"Training set: {len(train_df)}, Validation set: {len(val_df)}")

    # Step 4: Tokenize and pad sequences
    print("Tokenizing and padding sequences...")
    input_padded, output_padded, val_input_padded, val_output_padded, input_tokenizer, output_tokenizer = tokenize_and_pad(train_df, val_df)
    input_vocab_size = len(input_tokenizer.word_index) + 1
    output_vocab_size = len(output_tokenizer.word_index) + 1
    print(f"Input Vocabulary Size: {input_vocab_size}, Output Vocabulary Size: {output_vocab_size}")

    # Step 5: Build Seq2Seq model
    print("Building Seq2Seq model...")
    seq2seq_model, encoder_states = build_seq2seq_model(input_vocab_size, output_vocab_size)
    print("Seq2Seq model built.")

    # Step 6: Train the model
    print("Compiling and training the model...")
    compile_and_train_model(seq2seq_model, input_padded, output_padded, val_input_padded, val_output_padded, train_df,
                            save_path=model_save_path)
    print("Model training complete.")

    # Step 7: Build encoder and decoder models
    print("Building encoder and decoder models...")
    # Build encoder model
    encoder_model = tf.keras.Model(seq2seq_model.input[0], encoder_states)

    # Build decoder model
    decoder_state_input_h = tf.keras.Input(shape=(512,))
    decoder_state_input_c = tf.keras.Input(shape=(512,))
    decoder_inputs = tf.keras.Input(shape=(None,))
    decoder_embedding_layer = seq2seq_model.get_layer(index=3)  # Assuming the embedding is the 4th layer
    decoder_embedding = decoder_embedding_layer(decoder_inputs)

    decoder_lstm = seq2seq_model.get_layer(index=4)  # Assuming the LSTM is the 5th layer
    decoder_dense = seq2seq_model.get_layer(index=-1)  # Assuming the Dense layer is the last layer
    decoder_lstm_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=[decoder_state_input_h, decoder_state_input_c]
    )
    decoder_outputs = decoder_dense(decoder_lstm_outputs)
    decoder_model = tf.keras.Model(
        [decoder_inputs, decoder_state_input_h, decoder_state_input_c],
        [decoder_outputs, state_h, state_c]
    )
    print("Encoder and decoder models built.")

    # Step 8: Perform inference
    print("Performing inference...")
    test_input = sequenceDataInput(genRandomUserInput(exercisesDF, equipmentDF))
    print("Test input:", test_input)
    test_output = infer(
        test_input,
        encoder_model,
        decoder_model,
        input_tokenizer,
        output_tokenizer,
        input_padded.shape[1]
    )
    print("Inference output:", test_output)

exercise, muscles, equipment = fullProcessData()
# Run the workflow
#run_workflow()
results = generateTrainingDataFrame(exercisesDataFrame=exercise,musclesDataFrame=muscles,equipmentDataFrame=equipment,numSamples=10)