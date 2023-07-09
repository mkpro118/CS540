################################################################################
# Code Attribution: Module neural_network is my own work.                      #
#                   Source code for this module can be found at                #
#                   https://github.com/mkpro118/neural_network                 #
################################################################################

################################################################################
# No other part of this source code has been procured from anywhere except the #
#    python standard library and the official docs of numpy and matplotlib     #
################################################################################


import numpy as np
import json
from typing import Union
from neural_network.models import Sequential

CONFIG_FILE_PART_1 = 'p1_part1_model.json'
CONFIG_FILE_PART_2 = 'p1_part2_model.json'
TRAINING_DATA_SOURCE = 'mnist_train_p1.csv'
TESTING_DATA_SOURCE = 'test.txt'


def get_data(
        filename: str,
        has_labels: bool = True,
        label_index: int = 0) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    data = np.genfromtxt(filename, delimiter=',')

    if has_labels:
        return np.delete(data, label_index, axis=1), data[:, label_index]

    return data


def preprocess_data(X: np.ndarray) -> np.ndarray:
    return X / 255


def create_model(part: int) -> Sequential:
    from neural_network.layers import Dense

    model = Sequential()

    if part == 1:
        model.add(Dense(1, input_shape=784, activation='sigmoid'))
    elif part == 2:
        model.add(
            Dense(28, input_shape=784, activation='sigmoid'),
            Dense(1, activation='sigmoid')
        )
    else:
        raise ValueError(
            f'{part = } is not a valid part, it should be either 1 or 2')

    model.compile(cost='mse', metrics=['accuracy'])

    return model


def train_model(
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: bool = True) -> Sequential:
    print('creating new model')
    from neural_network.model_selection import KFold

    kf = KFold(n_splits=4)
    print(len(X_train))

    for train, validate in kf.split(X_train):
        model.fit(
            X_train[train],
            y_train[train],
            epochs=100,
            validation_data=(X_train[validate], y_train[validate]),
            verbose=verbose
        )
        break

    return model


def load_model(config_file: str) -> Sequential:
    with open(config_file) as config:
        return Sequential.build_from_config(json.load(config))


def get_or_create_trained_model(
        config_file: str,
        data_source: str,
        part: int) -> Sequential:
    try:
        return load_model(config_file)
    except Exception:
        X, y = get_data(data_source)
        return train_model(create_model(part), X, y)


def get_weights(model: Sequential, layer: int = -1) -> np.ndarray:
    return np.around(model.layers[layer].weights, 4)


def get_bias(model: Sequential, layer: int = -1) -> np.ndarray:
    return np.around(model.layers[layer].bias, 4)


def save_data(filename: str, data: str):
    with open(filename, 'w') as f:
        f.write(data)


def question_1():
    save_data(
        'q1.txt',
        ','.join(
            map(
                lambda x: f'{x:0.2f}',
                preprocess_data(
                          get_data(TRAINING_DATA_SOURCE)[0][0]).tolist()
            )
        )
    )


def question_2():
    save_data(
        'q2.txt',
        ','.join(
            map(
                lambda x: f'{x:0.4f}',
                get_weights(
                    (model := get_or_create_trained_model(
                        CONFIG_FILE_PART_1,
                        TRAINING_DATA_SOURCE,
                        1
                    ))
                ).flatten().tolist()
            )
        ) + f',{get_bias(model).reshape(())}'
    )


def question_3():
    save_data(
        'q3.txt',
        ','.join(
            map(
                lambda x: f'{x:0.2f}',
                np.around(
                    np.reshape(
                        get_or_create_trained_model(
                            CONFIG_FILE_PART_1,
                            TRAINING_DATA_SOURCE,
                            1
                        ).predict(
                            preprocess_data(
                                get_data(
                                    TESTING_DATA_SOURCE,
                                    has_labels=False
                                )
                            )
                        ),
                        (-1,)
                    ), 2).tolist()
            )
        )
    )


def question_4():
    save_data(
        'q4.txt',
        ','.join(
            map(
                str,
                (np.reshape(
                    get_or_create_trained_model(
                        CONFIG_FILE_PART_1,
                        TRAINING_DATA_SOURCE,
                        1
                    ).predict(
                        get_data(
                            TESTING_DATA_SOURCE,
                            has_labels=False
                        )
                    ),
                    (-1,)
                ) >= 0.5
                ).astype(int).tolist()
            )
        )
    )


def question_5():
    save_data(
        'q5.txt',
        '\n'.join(
            map(
                lambda x: ','.join(
                    map(lambda x: f'{x:0.4f}', x)
                ),
                get_weights(
                    (model := get_or_create_trained_model(
                        CONFIG_FILE_PART_2,
                        TRAINING_DATA_SOURCE,
                        2
                    )),
                    layer=0
                ).reshape((784, 28)).tolist()
            )
        ) + '\n' + ",".join(
            map(
                lambda x: f"{x:0.4f}",
                get_bias(model, layer=0)
                .flatten()
                .tolist()
            )
        )
    )


def question_6():
    save_data(
        'q6.txt',
        ','.join(
            map(
                lambda x: f'{x:0.4f}',
                get_weights(
                    (model := get_or_create_trained_model(
                        CONFIG_FILE_PART_2,
                        TRAINING_DATA_SOURCE,
                        2
                    ))
                ).flatten().tolist()
            )
        ) + f',{get_bias(model).reshape(())}'
    )


def question_7():
    save_data(
        'q7.txt',
        ','.join(
            map(
                lambda x: f'{x:0.2f}',
                np.around(
                    np.reshape(
                        get_or_create_trained_model(
                            CONFIG_FILE_PART_2,
                            TRAINING_DATA_SOURCE,
                            2
                        ).predict(
                            preprocess_data(
                                get_data(
                                    TESTING_DATA_SOURCE,
                                    has_labels=False
                                )
                            )
                        ),
                        (-1,)
                    ), 2).tolist()
            )
        )
    )


def question_8():
    save_data(
        'q8.txt',
        ','.join(
            map(
                str,
                (np.reshape(
                    get_or_create_trained_model(
                        CONFIG_FILE_PART_1,
                        TRAINING_DATA_SOURCE,
                        1
                    ).predict(
                        get_data(
                            TESTING_DATA_SOURCE,
                            has_labels=False
                        )
                    ),
                    (-1,)
                ) >= 0.5
                ).astype(int).tolist()
            )
        )
    )


def question_9():
    model = get_or_create_trained_model(
        CONFIG_FILE_PART_2,
        TRAINING_DATA_SOURCE,
        2
    )

    X, y = get_data(TRAINING_DATA_SOURCE)
    X = preprocess_data(X)
    y = y.flatten()

    predictions = (model.predict(X) >= 0.5).astype(int).flatten()
    incorrect = np.argwhere(predictions != y)[0]

    save_data('q9.txt', ','.join(
        map(lambda x: f'{x:0.2f}', X[incorrect].flatten().tolist())))

    # # To plot incorrect sample
    # from matplotlib import pyplot as plt
    # plt.imshow(np.reshape(X[incorrect], (28, 28)), cmap='gray')
    # plt.show(block=True)


def main():
    # question_1()
    # print('Question 1 done!')
    question_2()
    print('Question 2 done!')
    # question_3()
    # print('Question 3 done!')
    # question_4()
    # print('Question 4 done!')
    question_5()
    print('Question 5 done!')
    # question_6()
    # print('Question 6 done!')
    # question_7()
    # print('Question 7 done!')
    # question_8()
    # print('Question 8 done!')
    # question_9()
    # print('Question 9 done!')


if __name__ == '__main__':
    main()
