import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

def emoticonModel(train_file, test_file):
    # Load training and test data
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Function to prepare data by splitting emoticons
    def prepare_data(data):
        emoticon_columns = data['input_emoticon'].apply(lambda x: list(x))
        return emoticon_columns
        
    # Prepare the input data
    X_train = prepare_data(train_data)
    X_test = prepare_data(test_data)

    y_train = train_data['label']

    # Tokenization
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(X_train)

    # Vocabulary size
    vocab_size = len(tokenizer.word_index) + 1

    # Padding sequences
    max_len = 13  # As we have 13 emoticon positions
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')

    # Assign y values directly (already binary)
    y_train_enc = y_train

    # Model hyperparameters (fixed)
    lstm_units = 56
    dense_units = 56
    dropout_rate = 0.3

    # Build the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=8, input_length=max_len))
    model.add(SimpleRNN(lstm_units, return_sequences=False))  # SimpleRNN layer
    model.add(Dropout(dropout_rate))
    model.add(Dense(dense_units, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Assuming binary classification

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_pad, y_train_enc, epochs=10, batch_size=32, verbose=1)

    # Predict on test set
    y_pred = model.predict(X_test_pad)
    y_pred_classes = np.round(y_pred).astype(int)

    return y_pred_classes.flatten()  # Return predicted classes

def deepFeaturesModel(train_file, test_file):
    # Load training feature dataset
    train_feat = np.load(train_file, allow_pickle=True)
    train_feat_X = train_feat['features']
    train_feat_Y = train_feat['label']

    # Load test dataset
    test_feat_X = np.load(test_file, allow_pickle=True)['features']

    # Convert labels to integers if they are not already
    train_feat_Y = train_feat_Y.astype(int)  # Assuming labels are 0 and 1

    # Reshape the data from (n_samples, 13, 786) to (n_samples, 13 * 786)
    n_train_samples = train_feat_X.shape[0]
    n_test_samples = test_feat_X.shape[0]

    X_train = train_feat_X.reshape(n_train_samples, -1)  # Flatten to (n_samples, 13 * 786)
    X_test = test_feat_X.reshape(n_test_samples, -1)  # Flatten to (n_samples, 13 * 786)

    # Standardize the training and validation features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train the Random Forest model using 100% of the dataset
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, train_feat_Y)

    # Make predictions on the validation set
    y_test_pred = model.predict(X_test)

    return y_test_pred  # Return predicted values

def textSeqModel(train_file, test_file):
    # Load and preprocess the datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    substrings_to_remove = ['15436', '1596', '464', '614', '262', '422', '284']
    
    # Function to remove the specified substrings
    def remove_substrings(sequence, substrings):
        for substring in substrings:
            sequence = sequence.replace(substring, '')  # Remove each substring
        return sequence

    # Preprocess the data
    # def preprocess_data(df):
    #     df['processed_str'] = df['input_str'].apply(lambda x: x[3:])  # Remove leading zeros
    #     df['modified_str'] = df['processed_str'].apply(lambda x: remove_substrings(x, substrings_to_remove))
    #     df['modified_length'] = df['modified_str'].apply(len)
    #     df = df[df['modified_length'] == 13]  # Keep only strings with length 13 after modification
    #     return df
    
    train_df['processed_str'] = train_df['input_str'].apply(lambda x: x[3:])  # Remove leading zeros
    test_df['processed_str'] = test_df['input_str'].apply(lambda x: x[3:])  # Remove leading zeros

    train_df['modified_str'] = train_df['processed_str'].apply(lambda x: remove_substrings(x, substrings_to_remove))
    train_df['modified_length'] = train_df['modified_str'].apply(len)
    train_df = train_df[train_df['modified_length'] == 13]

    test_df['modified_str'] = test_df['processed_str'].apply(lambda x: remove_substrings(x, substrings_to_remove))
    test_df['modified_length'] = test_df['modified_str'].apply(len)
    
    # # Preprocess train and test data
    # train_df = preprocess_data(train_df)
    # test_df = preprocess_data(test_df)

    # Convert the characters in modified_str to integers (character encoding)
    def encode_strings(df):
        all_chars = sorted(list(set("".join(df['modified_str'].values))))  # Get unique characters
        char_to_int = {char: i + 1 for i, char in enumerate(all_chars)}  # Map each character to an integer
        df['encoded_str'] = df['modified_str'].apply(lambda x: [char_to_int[char] for char in x])  # Encode each string
        return df, char_to_int

    train_df, char_to_int = encode_strings(train_df)
    test_df, _ = encode_strings(test_df)

    # Prepare data for XGBoost
    # We will flatten the encoded strings into features
    X_train = pd.DataFrame(train_df['encoded_str'].tolist())
    y_train = train_df['label'].values

    X_test = pd.DataFrame(test_df['encoded_str'].tolist())

    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Predict on the test data
    y_test_pred = model.predict(X_test)

    return y_test_pred  # Return predicted values for test data

def combinedModel(emoticon_train_file, seq_train_file, feature_train_file, test_emoticon_file, test_seq_file, test_feature_file):
    # Load the training datasets
    train_emoticon_df = pd.read_csv(emoticon_train_file)
    train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
    train_emoticon_Y = train_emoticon_df['label'].tolist()

    train_seq_df = pd.read_csv(seq_train_file)
    train_seq_X = train_seq_df['input_str'].tolist()
    train_seq_Y = train_seq_df['label'].tolist()

    train_feat = np.load(feature_train_file, allow_pickle=True)
    train_feat_X = train_feat['features']
    train_feat_Y = train_feat['label']

    # Load the test datasets
    test_emoticon_df = pd.read_csv(test_emoticon_file)
    test_emoticon_X = test_emoticon_df['input_emoticon'].tolist()

    test_seq_df = pd.read_csv(test_seq_file)
    test_seq_X = test_seq_df['input_str'].tolist()

    test_feat = np.load(test_feature_file, allow_pickle=True)
    test_feat_X = test_feat['features']

    # OneHot Encoding for Emoticon and Text Sequence Datasets
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')

    # One-Hot Encode Emoticon Dataset (Training)
    train_emoticon_encoded = onehot_encoder.fit_transform(np.array(train_emoticon_X).reshape(-1, 1)).toarray()

    # One-Hot Encode Text Sequences Dataset (Training)
    train_seq_encoded = onehot_encoder.fit_transform(np.array(train_seq_X).reshape(-1, 1)).toarray()

    # One-Hot Encode Emoticon Dataset (Test)
    test_emoticon_encoded = onehot_encoder.transform(np.array(test_emoticon_X).reshape(-1, 1)).toarray()

    # One-Hot Encode Text Sequences Dataset (Test)
    test_seq_encoded = onehot_encoder.transform(np.array(test_seq_X).reshape(-1, 1)).toarray()

    # Scale the Feature Matrix (Feature Dataset)
    scaler = StandardScaler()

    # Scale feature matrix for training data
    train_feat_scaled = scaler.fit_transform(train_feat_X.reshape(train_feat_X.shape[0], -1))

    # Scale feature matrix for test data
    test_feat_scaled = scaler.transform(test_feat_X.reshape(test_feat_X.shape[0], -1))

    # Concatenate all encoded/processed datasets
    train_X_combined = np.hstack((train_emoticon_encoded, train_seq_encoded, train_feat_scaled))
    train_Y_combined = np.array(train_emoticon_Y)  # Use emoticon labels for training

    test_X_combined = np.hstack((test_emoticon_encoded, test_seq_encoded, test_feat_scaled))

    # Train an SVM Classifier
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(train_X_combined, train_Y_combined)

    # Make predictions on the test set
    test_pred = svm_classifier.predict(test_X_combined)

    return test_pred  # Return predicted values for test data



# Helper function to save predictions to a file
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")


if __name__ == '__main__':
    # Load the test datasets
    test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']
    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
    
    
    # Make predictions using the trained models
    pred_emoticons = emoticonModel('datasets/train/train_emoticon.csv', 'datasets/test/test_emoticon.csv')
    pred_feat = deepFeaturesModel('datasets/train/train_feature.npz', 'datasets/test/test_feature.npz')
    pred_text = textSeqModel('datasets/train/train_text_seq.csv', 'datasets/test/test_text_seq.csv')
    pred_combined = combinedModel(
    "datasets/train/train_emoticon.csv", 
    "datasets/train/train_text_seq.csv", 
    "datasets/train/train_feature.npz", 
    "datasets/test/test_emoticon.csv", 
    "datasets/test/test_text_seq.csv", 
    "datasets/test/test_feature.npz"
)
    
    # Save predictions to text files
    save_predictions_to_file(pred_feat, "pred_feat.txt")
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    save_predictions_to_file(pred_text, "pred_text.txt")
    save_predictions_to_file(pred_combined, "pred_combined.txt")
