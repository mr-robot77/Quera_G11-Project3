# Load Data
* Create DataFrame for **Train Data**
    | image_dir                                           | label |
    |-----------------------------------------------------|-------|
    | /kaggle/input/image-prooduct-quera/train_data/...   | 7     |
    | /kaggle/input/image-prooduct-quera/train_data/...   | 7     |
    | /kaggle/input/image-prooduct-quera/train_data/...   | 7     |
    | /kaggle/input/image-prooduct-quera/train_data/...   | 7     |
    | /kaggle/input/image-prooduct-quera/train_data/...   | 7     |

* Create DataFrame for **Test Data**
    | image_dir                                           |
    |-----------------------------------------------------|
    | /kaggle/input/image-prooduct-quera/test_data/t...   |
    | /kaggle/input/image-prooduct-quera/test_data/t...   |
    | /kaggle/input/image-prooduct-quera/test_data/t...   |
    | /kaggle/input/image-prooduct-quera/test_data/t...   |
    | /kaggle/input/image-prooduct-quera/test_data/t...   |

* Check Number of Samples in each Class
    * ![Alt text](images/P1/image_classification_num_sample_class.PNG)

* Plot some Images in each Class
    * ![Alt text](images/P1/some_images.png)

# Define Functions
## Function for Create Generators of Train | Validation | Test
```python
def create_generators(data, 
                      test_size, 
                      preprocessing_function,
                      batch_size = 32,
                      image_size = (224, 224)
                     ):
    
    # Split Data to Tain and Validation
    train, validation = train_test_split(data, test_size=test_size, random_state=42)
    train = train.reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    
    # Create Data Generators
    train_data_generator = ImageDataGenerator(
    preprocessing_function=preprocessing_function,
    rotation_range=15, 
#     width_shift_range=0.2,  
#     height_shift_range=0.2, 
    brightness_range=[0.8,1.2],
    shear_range=0.2,  
#     zoom_range=0.2, 
    horizontal_flip=True,
#     vertical_flip=True,
    fill_mode='nearest' 
    )
    
    test_data_generator = ImageDataGenerator(
            preprocessing_function=preprocessing_function
        )

    train_generator = train_data_generator.flow_from_dataframe(
        dataframe= train,
        x_col='image_dir',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', 
        shuffle=True,
        seed=42
    )
    validation_generator = test_data_generator.flow_from_dataframe(
        dataframe= validation,
        x_col='image_dir',
        y_col='label',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical', 
        shuffle=False
    )

    # test_generator = test_data_generator
    
    return train_generator, validation_generator, test_data_generator
```
## Function for Data Augmentation on each Bach
```python
def augment_batch(x_batch, y_batch, data_gen: ImageDataGenerator, n=2):
    augmented_images = []
    augmented_labels = []
    
    for x, y in zip(x_batch, y_batch):
        
        augmented_images.append(x)
        augmented_labels.append(y)
        
        for i in range(1,n):
            x_aug = data_gen.random_transform(x)
            augmented_images.append(x_aug)
            augmented_labels.append(y)
    
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    indices = np.arange(augmented_images.shape[0])
    np.random.shuffle(indices)
    augmented_images = augmented_images[indices]
    augmented_labels = augmented_labels[indices]
    
    return augmented_images, augmented_labels

def custom_generator(generator, data_gen, n=2):
    for x_batch, y_batch in generator:
        x_batch_aug, y_batch_aug = augment_batch(x_batch, y_batch, data_gen, n)
        yield x_batch_aug, y_batch_aug
```
## Function for Create pretrained Model as Base Model
```python
def make_base_model(model=ResNet50,
                    image_size=(224,224),
                    num_classes=10,
                    num_trainable_layeres=0,
                    kaggle=False,
                    weights_path=None,
                   ):
    
    # base_model
    if kaggle:
        # Kaggle
        base_model = model(
        input_shape=(image_size[0], image_size[1], 3),
        classes=num_classes,
        include_top=False, 
        weights=None,
        pooling=None #'avg'
        )
        base_model.load_weights(weights_path)
    else:
        # Colab
        base_model = model(
        input_shape=(image_size[0], image_size[1], 3),
        classes=num_classes,
        include_top=False, 
        weights='imagenet',
        pooling=None #'avg'
        )
    for layer in base_model.layers:
        layer.trainable = False
    if num_trainable_layeres > 0:
        for layer in base_model.layers[-num_trainable_layeres : ]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True    
    
    return base_model
```
## Function for Add our Layeres to Base Model
```python
def try_model(base_model, 
              n_added_layers=2, 
              filters=[128,128], 
              activations=['relu', 'relu'],
              regularize=False,
              augmentation_layer=False,
              num_classes=10,
              optimizer=Adam(learning_rate=1e-3),
              metrics=['accuracy']):
    if augmentation_layer:
        data_augmentation = keras.Sequential(
        [
        layers.RandomRotation(0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
        ]
        )
        inputs = base_model.input
        x = data_augmentation(inputs)
#         scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
#         x = scale_layer(x)
        x = base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x) 
        
    else:
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x) 
        
    for i in range(n_added_layers):
        if regularize:
            x = Dense(filters[i], activation='relu', kernel_regularizer=l2(0.01))(x)
        else:
            x = Dense(filters[i], activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    trainable_params = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_params = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=metrics)
#     model.summary()
    return model
```
## Functions for Evaluation Metrics
```python
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def f1_micro(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    actual_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (actual_positives + K.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    
    return f1
```
## Functions for Save and Load Models
```python
def save_model(model, filename, path=''):
    filename_json = os.path.join(path, filename + '.json')
    filename_h5 = os.path.join(path, filename + '.h5')

    # serialize model to JSON
    model_json = model.to_json()
    with open(filename_json, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(filename_h5)
    print("Saved model to ", filename_h5 )
    
def load_model(model, filename, path=''):
    filename_json = os.path.join(path, filename + '.json')
    filename_h5 = os.path.join(path, filename + '.h5')
    
    # load json and create model
    json_file = open(filename_json , 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights(filename_h5)
    return loaded_model
```
# Try Models
## ResNet 50
* ResNet 50 | Train 0 Layers | LR = 1e-3
* ResNet 50 | Train 0 Layers | 2 Dense Layer 128 | LR = 1e-3
* ResNet 50 | Train 5 Layers | 2 Dense Layer 128 | LR = 1e-3
* ResNet 50 | Train 10 layers | 2 Dense Layer 256,128 | LR = 1e-3
* ResNet 50 | Train 15 layers | 2 Dense Layer 256,128 | LR = 1e-3
* ResNet 50 | Train 20 layers | 2 Dense Layer 256,128 | LR = 1e-4
* ResNet 50 | Train 25 layers | 2 Dense Layer 256,128 | LR = 1e-4
* ResNet 50 | Train 45 layers | 2 Dense Layer 256,128 | LR = 1e-4
    ```python
    batch_size = 32
    image_size = (224, 224)
    
    train_generator, validation_generator, test_data_generator = create_generators(data, 
                          test_size= 0.3, 
                          preprocessing_function= resnet50_preprocess_input,
                          batch_size = batch_size,
                          image_size = image_size
                         )
    
    resnet50_weights_path = '/kaggle/input/resnet50-weights-notop/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    base_model = make_base_model(model=ResNet50,
                        image_size=image_size,
                        num_classes=len(class_names),
                        num_trainable_layeres=45,
                        kaggle=kaggle,
                        weights_path=resnet50_weights_path,
                       )
    
    model = try_model(base_model, 
                  n_added_layers= 2, 
                  filters= [256,128], 
                  activations= ['relu', 'relu'], 
                  num_classes= len(class_names),
                  optimizer= Adam(learning_rate=1e-4),
                  metrics= ['accuracy', f1_micro])
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_f1_micro',
                                                   min_delta=0,
                                                   patience=5,
                                                   mode='max',
                                                   restore_best_weights=True)
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=100,
        callbacks=[early_stopping]
    )
    
    # save model
    save_model(model, 
               filename='resnet50_train_45_add_2_layers_256_128_lr-4', 
               path=path)
    ```

## Xception
## EfficientNet V2 M

# Choose Best Model
<font color="red">این متن رنگ قرمز است</font>
* **Model:** Efficient V2 M
* **Trainable Layers:** Train 20 Layers
* **Added Layers:** 2 Dense Layers 256, 128
* **Learning Rate:** 1e-4
## Evaluation on validation
### Model with L2
### Model without L2
## Plot First Bach Validation Data with True,Predicted Label

## Test and Save Results

![Alt text](images/P1/image_classification.PNG)
![Alt text](images/P1/image_classification_num_sample_class.PNG)
![Alt text](images/P1/image_classification_val_Model_L2.PNG)
![Alt text](images/P1/image_classification_val_Model_no_L2.PNG)
![Alt text](images/P1/image_classification_test_Model_L2.PNG)
![Alt text](images/P1/image_classification_test_Model_no_L2.PNG)
