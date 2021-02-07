import tensorflow as tf

def get_checkpoints():
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
        baseline=None, restore_best_weights=False
    )

    save_best_model = tf.keras.callbacks.ModelCheckpoint(
        CPKT, monitor='val_loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='auto', save_freq='epoch', options=None
    )

    return [early_stop, save_best_model]