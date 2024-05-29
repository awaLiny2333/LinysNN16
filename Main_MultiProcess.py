import math
import random
import copy
import time
import multiprocessing


def random_params():
    random_start = []
    for i in range(197*16+17*16+17*10):
        random_start.append(random.randint(-5, 5))
    return random_start


DATASET = open('dataset.txt', 'r').read().split("\n")

PARAMS = open('params.txt', 'r').read().split("\n")
# PARAMS = random_params()
PARAMS = [float(awa) for awa in PARAMS]

layer_i_1 = []
layer_1_2 = []
layer_2_o = []

data_1 = []
data_2 = []
data_o = []


def sigmoid(input_value: float):
    """

    :param input_value: The float
    :return: A value from 0 ~ 1
    """
    # print(input_value)
    if input_value > 1000:
        return 1
    if input_value < -1000:
        return 0
    return 1 / (1 + (math.pow(math.e, -0.1 * input_value)))


def load_params(params: list[float]):
    """

    :param params: The big list of a total of 1738 floats
    """

    global layer_i_1
    global layer_1_2
    global layer_2_o
    layer_i_1 = params[0:3152]
    layer_1_2 = params[3152:3424]
    layer_2_o = params[3424:3594]


def recognize(params: list[float], input_layer: list[float]):
    """

    :param params: The Params
    :param input_layer: A plain list of 196 floats
    :return: [result_sorted_by_possibility, data_of_output_layer]
    """

    load_params(params)

    # print(input_layer)
    input_layer = [float(awa) for awa in input_layer]

    global data_o
    global data_1
    global data_2
    global layer_i_1
    global layer_1_2
    global layer_2_o

    data_1 = []
    data_2 = []
    data_o = []

    for out in range(16):
        neuron_sum = 0
        neuron_weight = layer_i_1[out * 197:(out + 1) * 197 - 1]
        neuron_weight = [float(awa) for awa in neuron_weight]

        neuron_bias = float(layer_i_1[(out + 1) * 197 - 1])
        for inner in range(196):
            neuron_sum += neuron_weight[inner] * input_layer[inner]
        data_1.append(sigmoid(neuron_sum + neuron_bias))

    for out in range(16):
        neuron_sum = 0
        neuron_weight = layer_1_2[out * 17:(out + 1) * 17 - 1]
        neuron_weight = [float(awa) for awa in neuron_weight]

        neuron_bias = float(layer_1_2[(out + 1) * 17 - 1])
        for inner in range(16):
            neuron_sum += neuron_weight[inner] * data_1[inner]
        data_2.append(sigmoid(neuron_sum + neuron_bias))

    for out in range(10):
        neuron_sum = 0
        neuron_weight = layer_2_o[out * 17:(out + 1) * 17 - 1]
        neuron_weight = [float(awa) for awa in neuron_weight]

        neuron_bias = float(layer_2_o[(out + 1) * 17 - 1])
        for inner in range(16):
            neuron_sum += neuron_weight[inner] * data_2[inner]
        data_o.append(sigmoid(neuron_sum + neuron_bias))

    result_sorted = []
    sort_temp = copy.deepcopy(data_o)
    for k in range(10):
        sort_max = 0
        for a in range(len(sort_temp)):
            if sort_temp[a] > sort_temp[sort_max]:
                sort_max = a
        result_sorted.append(sort_max)
        sort_temp[sort_max] = -1

    return [result_sorted, data_o]


def loss_acc_of(params: list[float], data_slice: str):
    """

    :param params: The Params
    :param data_slice: The slice of data (a line of str) from a set containing the target_digit in the first item
    :return: [loss, is_accurate]
    """

    # Filter out target_digit and real_data
    data_sliced = data_slice.split(" ")
    target_digit = int(data_sliced[0])
    real_data = [float(awa) for awa in data_sliced[1:197]]

    # Generates expect_loss
    expect_loss = []
    for k in range(10):
        if k == target_digit:
            expect_loss.append(1)
        else:
            expect_loss.append(0)

    # Get recognize result
    result = recognize(params, real_data)
    result_acc = result[0][0] == target_digit  # See if it's correct or not
    result_loss = result[1]

    # Count total loss
    loss = 0
    for item in range(10):
        loss += math.pow((expect_loss[item] - result_loss[item]), 2)

    return [loss, result_acc]


def loss_acc_avg_of_set(params: list[float], dataset: list[str]):
    """

    :param params: The Params
    :param dataset: The whole set of lines of data in str
    :return: A list 2 floatings: [average_loss, accuracy]
    """

    total_loss = 0
    total_acc = 0
    dataset_size = len(dataset)

    for i in range(dataset_size):
        l_a = loss_acc_of(params, dataset[i])
        total_loss += l_a[0]
        total_acc += l_a[1]

    return [total_loss / dataset_size, total_acc / dataset_size]


def get_gradients(params: list[float], dataset: list[str], step: float, start: int, end: int):
    """
    THIS Consumes most of the time and performance :P
    Consider improving this with multiprocessing
    :param params: The Params
    :param dataset: The Dataset
    :param step: The value of unit increment
    :param start: The start of gradient list (inclusive)
    :param end: The end of gradient list (exclusive)
    :return: The list of rough gradients
    """

    original_loss = loss_acc_avg_of_set(params, dataset)[0]

    params_size = len(params)
    loss_list = []

    for i in range(start, (end if end < params_size else params_size)):
        adjusted_params = copy.deepcopy(params)
        adjusted_params[i] = adjusted_params[i] + step
        gradient = loss_acc_avg_of_set(adjusted_params, dataset)[0] - original_loss
        loss_list.append(gradient)

    print(str(start) + "-" + str(end if end < params_size else params_size), end=" ")

    return loss_list


def get_gradients_uni(inputs):
    # print(inputs[3], inputs[4])
    # print(inputs[5], end=" ")
    return get_gradients(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])


def get_gradients_multiprocess(params: list[float], dataset: list[str], step: float):
    pool = multiprocessing.Pool(20)

    all_inputs = []
    for i in range(20):
        all_inputs.append([params, dataset, step, i*180, (i+1)*180, i])

    print("[GRADIENTS] All processes started.")
    print("[GRADIENTS_OK]", end=" ")

    result = pool.map(get_gradients_uni, all_inputs)
    results = sum(result, [])

    pool.close()
    pool.join()

    return results


def adjust_params(params: list[float], list_gradients: list[float], step: float):
    """

    :param params: The Params
    :param list_gradients: The Gradients
    :param step: The value of each increment
    :return: The adjusted params
    """
    for i in range(len(list_gradients)):
        params[i] += -1 * step * gradients[i]
    return params


# print(recognize(PARAMS, DATASET[0].split()[1:197]))
# print(loss_acc_of(PARAMS, DATASET[0]))
# print(loss_acc_avg_of_set(PARAMS, DATASET))

if __name__ == '__main__':

    time_start = time.time()

    loss_and_acc = loss_acc_avg_of_set(PARAMS, DATASET)
    new_loss = loss_and_acc[0]
    new_acc = loss_and_acc[1]
    print("Starting from: ")
    print("[LOSS_] " + str(new_loss))
    print("[ACC_] " + str(new_acc))
    print("LINK START!\n")

    # Auto training for 1000 Steps

    step_of_get_gradients = 0.01
    for out_cycle in range(1000):

        gradients = get_gradients_multiprocess(PARAMS, DATASET, step_of_get_gradients)

        print("\n[GRADIENTS_Step] " + str(step_of_get_gradients))

        new_loss = 0
        new_acc = 0
        last_loss = 999
        in_cycle = 0

        for in_cycle in range(3000):
            PARAMS = adjust_params(PARAMS, gradients, 5)
            loss_and_acc = loss_acc_avg_of_set(PARAMS, DATASET)
            new_loss = loss_and_acc[0]
            new_acc = loss_and_acc[1]

            if in_cycle % 50 == 0:
                print("[LOSS_" + str(in_cycle) + "] " + str(new_loss), end=" ")
                print("[ACC_" + str(in_cycle) + "] " + str(new_acc))
            if new_loss > last_loss:
                PARAMS = adjust_params(PARAMS, gradients, -3)
                loss_and_acc = loss_acc_avg_of_set(PARAMS, DATASET)
                new_loss = loss_and_acc[0]
                new_acc = loss_and_acc[1]

                print("[LOSS_" + str(in_cycle) + "] " + str(new_loss), end=" ")
                print("[ACC_" + str(in_cycle) + "] " + str(new_acc) + "\n")

                if in_cycle < 20:
                    step_of_get_gradients = step_of_get_gradients * 0.5
                    print("[STEP] Step_of_get_gradients HALVED because of unlowerable loss :O → " +
                          str(step_of_get_gradients))

                break
            last_loss = new_loss

        if in_cycle > 2000:
            step_of_get_gradients = step_of_get_gradients * 2
            print("[STEP] Step_of_get_gradients DOUBLED because of currently too low speed :O → " +
                  str(step_of_get_gradients))

        fout = open('output/test/0_9_' + str(round(new_loss * 10000) / 10000) + '_' + str(new_acc) + '.txt', 'w')
        fout.write("\n".join([str(awa) for awa in PARAMS]))
        fout.close()

        print("[STEP] " + str(time.time() - time_start) + "s (" +
              str((time.time() - time_start) / 3600) + "h) Taken to reach the " + str(out_cycle) + "th Step\n")

# End of programme
