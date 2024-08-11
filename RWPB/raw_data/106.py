import numpy as np

def interpolate_features(features,
                         input_rate,
                         output_rate,
                         output_len):
    """
    Interpolate DeepSpeech features.

    Parameters
    ----------
    features : np.array
        DeepSpeech features.
    input_rate : int
        input rate (FPS).
    output_rate : int
        Output rate (FPS).
    output_len : int
        Output data length.

    Returns
    -------
    np.array
        Interpolated data.
    """
    # ----
    
    input_len = features.shape[0]
    num_features = features.shape[1]
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feature_idx in range(num_features):
        output_features[:, feature_idx] = np.interp(
            x=output_timestamps,
            xp=input_timestamps,
            fp=features[:, feature_idx])
    return output_features


# unit test cases
features = np.random.rand(10, 2)
input_rate = 2
output_rate = 2
output_len = 5
print(interpolate_features(features, input_rate, output_rate, output_len))

features = np.random.rand(10, 3)
input_rate = 5
output_rate = 5
output_len = 0
print(interpolate_features(features, input_rate, output_rate, output_len))

features = np.random.rand(15, 4)
input_rate = 3
output_rate = 4
output_len = 10
print(interpolate_features(features, input_rate, output_rate, output_len))