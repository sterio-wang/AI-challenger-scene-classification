import tensorflow as tf

def conv_network(feature, label, num_class, image_size, keep_prob):
    # Input layer
    input_layer = tf.reshape(feature, [-1, image_size, image_size, 3])

    # Conv1 
    conv1 = tf.layers.conv2d(
        inputs=input_layer, 
        filters=32, 
        kernel_size=3,
        strides=1, 
        padding='same', 
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer()
        )
    # Batch normalization 1
    bn1 = tf.layers.batch_normalization(conv1)
    pool1 = tf.layers.max_pooling2d(inputs=bn1, pool_size=[2, 2], strides=2)
    # Max pooling 1: [-1, image_size/2, image_size/2, 32]

    # Conv2
    conv2 = tf.layers.conv2d(
        inputs=pool1, 
        filters=64, 
        kernel_size=3,
        strides=1,
        padding='same', 
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer()
        )
    # Batch normalization 2
    bn2 = tf.layers.batch_normalization(conv2) 
    pool2 = tf.layers.max_pooling2d(inputs=bn2, pool_size=[2, 2], strides=2)
    # Max pooling 2: [-1, image_size/4, image_size/4, 64]

    # Conv3
    conv3 = tf.layers.conv2d(
        inputs=pool2, 
        filters=128, 
        kernel_size=3,
        strides=1,
        padding='same', 
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer()
        )
    # Batch normalization 2
    bn3 = tf.layers.batch_normalization(conv3)
    pool3 = tf.layers.max_pooling2d(inputs=bn3, pool_size=[2, 2], strides=2)
    # Max pooling 2: [-1, image_size/8, image_size/8, 128]

    # Flatten layer
    flatten = tf.reshape(pool3, [-1, image_size * image_size * 2]) 

    # Fully connected layer
    dense = tf.layers.dense(inputs=flatten, units=1024)
    dropout = tf.nn.dropout(dense, keep_prob) # or tf.layers.dropout(inputs, rate)

    # Output layer: returns logits and predictions
    logits = tf.layers.dense(dropout, units=num_class) 
    output = tf.sigmoid(logits)

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label))
    train_opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    return train_opt, cost, logits