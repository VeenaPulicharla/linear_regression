import csv


def read_data():
    with open('/Users/vpulicharla/Desktop/data.csv') as csvfile:
        data = []
        read_csv = csv.reader(csvfile, delimiter=',')
        for row in read_csv:
            data.append([row[0], row[1]])
        return data


def get_means(data_set):
    x_sum = 0
    y_sum = 0
    for i in data_set:
        x_sum += float(i[0])
        y_sum += float(i[1])
    x_mean = x_sum / len(data_set)
    y_mean = y_sum / len(data_set)
    return x_mean, y_mean


def get_slope(data, x_mean, y_mean):  # b1
    numerator_sum = 0
    denominator_sum = 0
    for i in data:
        numerator_sum += (float(i[0]) - float(x_mean)) * (float(i[0]) - float(y_mean))
        denominator_sum += (float(i[0]) - float(x_mean)) ** 2
    return float(numerator_sum) / float(denominator_sum)


def get_intercept(slope, x_mean, y_mean):  # b0
    return y_mean - (slope * x_mean)


def get_predicated_value(slope, intercept, independent_vars):
    for val in independent_vars:
        sum += val
    return ((sum * slope) + intercept)

def get_avg_error_rate(slope, intercept, data_points):
    error_rate = 0
    for i in data_points:
        actual_value = float(i[0])
        error_rate += (actual_value - get_predicated_value(slope, intercept, i[0])) ** 2
    return float(error_rate) / (float(len(data_points))*2)


def get_gradient_step_down_values(intercept, slope, data_points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    for i in data_points:
        b_gradient += -(2 / float(len(data_points))) * (float(i[1]) - (get_predicated_value(slope, intercept, i[0])))
        m_gradient += -(2 / float(len(data_points))) * float(i[0]) * (
                    float(i[1]) - (get_predicated_value(slope, intercept, i[0])))
    new_intercept = intercept - (learning_rate * b_gradient)
    new_gradient = slope - (learning_rate * m_gradient)
    return new_intercept, new_gradient


if __name__ == '__main__':
    # First load data into list of lists
    data_points = read_data()
    print("tests")
    # Get x and y means of given data set
    x_mean, y_mean = get_means(data_points)
    # Calculate slope using x and y means
    slope = get_slope(data_points, x_mean, y_mean)
    # Calculate intercept
    intercept = get_intercept(slope, x_mean, y_mean)
    print ("Error rate before Linear Regression without Gradient Descent approach")
    print(get_avg_error_rate(slope, intercept, data_points))
    print ("Error rate after Linear Regression with Gradient Descent approach")
    learning_rate = 0.0002
    for i in range(100000):
        intercept, slope = get_gradient_step_down_values(intercept, slope, data_points, learning_rate)
    print(get_avg_error_rate(slope, intercept, data_points))
