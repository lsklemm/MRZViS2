import numpy as np
import math
import functools

def activation(matrix):
    matrix = [[round(math.tanh(matrix[i][j])) for j in range(len(matrix[0]))] for i in range(len(matrix))]
    return matrix


def show_t(template):
    count = 1
    for i in range(len(template)):
        if count == 4:
            if template[i][0] == 1:
                print(1)
            else:
                print(0)
            count = 1
        else:
            if template[i][0] == 1:
                print(1, end=' ')
            else:
                print(0, end=' ')
            count += 1



def get_weights(filename):
    matrix = []
    with open(filename, 'r') as file:
        matrix = [[int(x) for x in line.split(' ')] for line in file]
    file.close()

    return matrix


def get_templates(file_name):
    templates = []
    template = []
    with open(file_name, 'r') as file:
        for line in file:
            if line == '\n':
                templates.append(template)
                template = []
            else:
                for x in line.split(' '):
                    if int(x) == 1:
                        template.append(1)
                    elif int(x) == 0:
                        template.append(-1)

        templates.append(template)

    file.close()
    return templates



def get_into_file(file, object):
    f = open(file, 'w')
    for i in range(len(object)):
        for j in range(len(object[0])):
            if j == len(object[0])-1:
                f.write((str(object[i][j])+'\n'))
            else:
                f.write((str(object[i][j]) + ' '))
    f.close()



def identify(template_file, weights_file):
    X = np.array(get_templates(template_file))
    weights = np.array(get_weights(weights_file))

    last_step = np.array(X).T
    relaxed = 0
    it = 0

    while relaxed != 1:
        it += 1
        if it > 1000:
            break
        s = np.matmul(weights, last_step)
        next_step = activation(s)

        show_t(last_step)
        print()

        if functools.reduce(lambda x, y: x and y, map(lambda a, b: a == b, last_step, next_step), True):
            print('Образ распознан')
            relaxed = 1

        it += 1
        last_step = next_step



def train(templates_file, weights_file):
    templates = get_templates(templates_file)

    weights = np.matmul(np.array([templates[0]]).T, np.array([templates[0]]))
    for i in range(1, len(templates)):
        w = (np.array(weights) + np.matmul(np.array([templates[i]]).T, np.array([templates[i]])))
        weights = w.tolist()


    for null_index in range(len(weights)):
        weights[null_index][null_index] = 0

    get_into_file(weights_file, weights)

    return weights


if __name__ == '__main__':
    print('1 - Обучение\n2 - Распознавание')
    user = int(input())
    if user == 1:
        print('Введите название файла с эталонными образами: ')
        templates_file = input()
        print('Введите название файла для матрицы весов: ')
        matrix_file = input()

        train(templates_file, matrix_file)
        print('Образы запомнены')


    elif user == 2:
        print('Введите название файла с матрицей весов: ')
        matrix_file = input()
        print('Введите название файла с искаженным образом: ')
        distorted_file = input()

        identify(distorted_file, matrix_file)