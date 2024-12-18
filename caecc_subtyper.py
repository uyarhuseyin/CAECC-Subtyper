import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture

class ConvolutionalAutoencoder:
    def __init__(self, omics, params=None):
        self.omics = self._scale_omics(omics)
        self.encoder = None
        self.model = None
        self.params = params if params else self._default_params()

    def _default_params(self):
        return {
            'first_fc_layer_nodes': 128,
            'first_conv_filter': 8,
            'second_conv_filter': 4,
            'bottleneck_units': 64,
            'activation': 'elu',
            'loss': 'mae',
            'apply_two_convs': True,
            'use_bottleneck_activation': True,
            'optimizer': 'adamax',
            'batch_size': 32
        }

    def _scale_omics(self, omics):
        scaler = MinMaxScaler()
        return [
            pd.DataFrame(scaler.fit_transform(omic), index=omic.index, columns=omic.columns)
            for omic in omics
        ]

    def _build_encoder(self, inputs):
        def process_fc_layers(input_layer):
            x = layers.Dense(self.params['first_fc_layer_nodes'], activation='linear')(input_layer)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(self.params['activation'])(x)
            x = tf.expand_dims(x, axis=2)
            return x

        # Apply dense encoding to reduce each input to the same size
        processed_layers = [process_fc_layers(input_layer) for input_layer in inputs]

        # Combine all processed layers to create image-like structure
        multiomics_img = layers.concatenate(processed_layers, axis=2)

        # Apply convolutional layers
        x = layers.Conv1D(self.params['first_conv_filter'], 1, padding='same')(multiomics_img)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation=self.params['activation'])(x)

        if self.params['apply_two_convs']:
            x = layers.Conv1D(self.params['second_conv_filter'], 1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation=self.params['activation'])(x)

        # Bottleneck layer
        x = layers.Flatten()(x)
        x = layers.Dense(self.params['bottleneck_units'])(x)

        if self.params['use_bottleneck_activation']:
            return layers.LeakyReLU(name='encoder')(x)
        return layers.Activation(self.params['activation'], name='encoder')(x)

    def _build_decoder(self, encoded):
        if self.params['apply_two_convs']:
            x = layers.Dense(self.params['first_fc_layer_nodes'] * self.params['second_conv_filter'])(encoded)
            x = layers.Reshape((self.params['first_fc_layer_nodes'], self.params['second_conv_filter']))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation=self.params['activation'])(x)
            x = layers.Conv1DTranspose(self.params['first_conv_filter'], 1, strides=1, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation=self.params['activation'])(x)
            x = layers.Conv1DTranspose(len(self.omics), 1, strides=1, padding='same')(x)
        else:
            x = layers.Dense(self.params['first_fc_layer_nodes'] * self.params['first_conv_filter'])(encoded)
            x = layers.Reshape((self.params['first_fc_layer_nodes'], self.params['first_conv_filter']))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation=self.params['activation'])(x)
            x = layers.Conv1DTranspose(len(self.omics), 1, strides=1, padding='same')(x)

        def create_output(y, output_shape, activation):
            y = layers.BatchNormalization()(y)
            y = layers.Activation(activation)(y)
            return layers.Dense(output_shape, activation=activation)(y)

        # Final decoding step: Apply dense layers to reconstruct omics data
        outputs = [create_output(x[:, :, i], omic.shape[1], self.params['activation']) for i, omic in enumerate(self.omics)]
        return outputs

    def build_model(self):
        # Define shared inputs
        inputs = [layers.Input(shape=(omic.shape[1])) for omic in self.omics]

        # Encoder
        encoded = self._build_encoder(inputs)

        # Decoder
        outputs = self._build_decoder(encoded)

        self.encoder = Model(inputs=inputs, outputs=encoded)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.params['optimizer'], loss=self.params['loss'], metrics=['mae', 'mse'])

    def train(self, epochs=50):
        self.model.fit(x=self.omics, y=self.omics, epochs=epochs, batch_size=self.params['batch_size'], verbose=1)

    def extract_features(self):
        extracted_features = self.encoder.predict(self.omics)
        return pd.DataFrame(extracted_features, index=self.omics[0].index)

class ConsensusClustering:
    @staticmethod
    def perform_clustering(k, features):
        # Scale features for clustering
        features_scaled = pd.DataFrame(StandardScaler().fit_transform(features), index=features.index)
      
        agg_labels = AgglomerativeClustering(n_clusters=k).fit_predict(features_scaled)
        kmeans_labels = KMeans(n_clusters=k).fit_predict(features_scaled)
        spec_labels = SpectralClustering(n_clusters=k).fit_predict(features_scaled)
        gaus_labels = GaussianMixture(n_components=k).fit_predict(features_scaled)

        def create_similarity_matrix(labels):
            n = len(labels)
            similarity_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if labels[i] == labels[j]:
                        similarity_matrix[i, j] = 1
            return similarity_matrix

        similarity_matrices = [
            create_similarity_matrix(agg_labels),
            create_similarity_matrix(spec_labels),
            create_similarity_matrix(gaus_labels),
            create_similarity_matrix(kmeans_labels),
        ]

        consensus_matrix = sum(similarity_matrices) / len(similarity_matrices)
        final_labels = AgglomerativeClustering(n_clusters=k).fit_predict(1 - consensus_matrix)

        return final_labels

class CAECCSubtyper:
    def __init__(self, omics):
        self.omics = self._preprocess_data(omics)
        self.autoencoder = ConvolutionalAutoencoder(self.omics)

    def _preprocess_data(self, omics):
        for omic in omics:
            omic.dropna(axis=1, how='any', inplace=True)

        intersection_of_index = omics[0].index
        for omic in omics[1:]:
            intersection_of_index = intersection_of_index.intersection(omic.index)

        return [omic.loc[intersection_of_index] for omic in omics]

    def train(self, epochs=50):
        self.autoencoder.build_model()
        self.autoencoder.train(epochs=epochs)

    def subtype(self, k):
        features = self.autoencoder.extract_features()
        clusters = ConsensusClustering.perform_clustering(k, features)
        return clusters

def load_omics_from_folder(folder_path):
    """Load all omics datasets from a specified folder."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Specified folder '{folder_path}' does not exist.")
    omics_data = []
    for file_name in sorted(os.listdir(folder_path)): 
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            omics_data.append(pd.read_csv(file_path, index_col=0))
    if not omics_data:
        raise ValueError(f"No CSV files found in folder '{folder_path}'.")
    return omics_data

def main():
    parser = argparse.ArgumentParser(description="CAECC-Subtyper Framework")
    parser.add_argument("--mode", type=str, required=True, choices=["features", "full_pipeline", "clustering"],
                        help="Mode of operation: 'features' for feature extraction only, 'full_pipeline' for feature extraction and clustering, 'clustering' for clustering only.")
    parser.add_argument("--output", type=str, required=True, help="Path to save results.")
    parser.add_argument("--data_folder", type=str, required=True, help="Path to the folder containing omics datasets in CSV format.")
    parser.add_argument("--clusters", type=int, default=5, help="Number of clusters for consensus clustering.")

    args = parser.parse_args()

    # Load omics data
    omics_data = load_omics_from_folder(args.data_folder)

    # Initialize CAECC-Subtyper
    subtyper = CAECCSubtyper(omics_data)

    if args.mode == "features":
        # Train convolutional autoencoder and extract features
        subtyper.train()
        features = subtyper.autoencoder.extract_features()
        os.makedirs(args.output, exist_ok=True)
        features.to_csv(os.path.join(args.output, "extracted_features.csv"))
        print(f"Features saved to {args.output}/extracted_features.csv")

    elif args.mode == "full_pipeline":
        # Train convolutional autoencoder, extract features, and perform clustering
        subtyper.train()
        features = subtyper.autoencoder.extract_features()
        os.makedirs(args.output, exist_ok=True)
        features.to_csv(os.path.join(args.output, "extracted_features.csv"))
        print(f"Features saved to {args.output}/extracted_features.csv")

        clusters = subtyper.subtype(args.clusters)
        pd.DataFrame({"Cluster": clusters}, index=features.index).to_csv(os.path.join(args.output, "clusters.csv"))
        print(f"Clusters saved to {args.output}/clusters.csv")

    elif args.mode == "clustering":
        # Only perform clustering (assumes extracted features are available)
        features_path = os.path.join(args.output, "extracted_features.csv")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Extracted features file not found at {features_path}. Run in 'features' or 'pipeline' mode first.")

        features = pd.read_csv(features_path, index_col=0)
        clusters = ConsensusClustering.perform_clustering(args.clusters, features)
        os.makedirs(args.output, exist_ok=True)
        pd.DataFrame({"Cluster": clusters}, index=features.index).to_csv(os.path.join(args.output, "clusters.csv"))
        print(f"Clusters saved to {args.output}/clusters.csv")

if __name__ == "__main__":
    main()
