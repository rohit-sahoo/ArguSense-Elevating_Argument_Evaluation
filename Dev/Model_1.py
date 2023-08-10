# Define a function to build the model
def build_model():
    # Define inputs for tokens and attention masks with the specified shape and data type
    tokens = tf.keras.layers.Input(shape=(MAX_LEN,), name='tokens', dtype=tf.int32)
    attention = tf.keras.layers.Input(shape=(MAX_LEN,), name='attention', dtype=tf.int32)

    # Load the model configuration from the downloaded model's config.json file
    config = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH + '/config.json')

    # Load the pre-trained model using the loaded configuration
    backbone = TFAutoModel.from_pretrained(DOWNLOADED_MODEL_PATH + '/tf_model.h5', config=config)

    # Pass the inputs through the backbone model
    x = backbone(tokens, attention_mask=attention)

    # Apply a dense layer with ReLU activation
    x = tf.keras.layers.Dense(256, activation='relu')(x[0])

    # Apply another dense layer with softmax activation for classification
    x = tf.keras.layers.Dense(15, activation='softmax', dtype='float32')(x)

    # Create the final model using inputs and outputs
    model = tf.keras.Model(inputs=[tokens, attention], outputs=x)

    # Compile the model with specified optimizer, loss function, and metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                  loss=[tf.keras.losses.CategoricalCrossentropy()],
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model