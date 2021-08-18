

__all__ = ['get_info_str', 'print_training_info']


def get_info_str(*key):
    """
    Returns a continuous string of a list of objects. Each object is transferred to a string and strings are separated
    by '_' to form the final string. If a string of an object is too long or the object is a class, it will be replaced
    by the class name of the object. All the string will be transferred to lower case.

    Args:
        *key: several objects

    Returns:
        information_string(str): a continuous string of the input objects.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            model = paddlefsl.vision.backbones.Conv4(input_size=(84, 84), output_size=5)
            ways = 5
            shots = 5
            info_str = utils.get_info_str(model, ways, 'ways', shots, 'shots') # info_str: 'conv4_5_ways_5_shots'

    """
    result = ''
    for k in key:
        s = str(k).lower()
        if '\n' in s or '<' in s:
            s = type(k).__name__.lower()
        if s == 'sequential':
            s = k._full_name
        if s != '':
            result += s + '_'
    return result[:-1]


def print_training_info(iteration,
                        train_loss,
                        train_acc=None,
                        valid_loss=None,
                        valid_acc=None,
                        report_file=None,
                        info=None):
    """
    Print training information in terminal or in file, including iteration number, training or validation loss and
    accuracy, and other information.

    Args:
        iteration(int): iteration number.
        train_loss(float): training loss.
        train_acc(float, optional): training accuracy, default None.
        valid_loss(float, optional): validation loss, default None.
        valid_acc(float, optional): validation accuracy, default None.
        report_file(str, optional): file path to which the information will be saved, default None. If None, this
            function will only print information in terminal. If not None, this function will add information to
            the tail of the file, so the file must exist before calling this function.
        info(object, optional): other information to print. This will only be printed in terminal after iteration.

    Returns:
        None. This function prints the information in terminal or save information in file.

    Examples:
        ..code-block:: python

            import paddlefsl.utils as utils
            train_loss, train_acc = 0.85, 0.76
            utils.print_training_info(0, train_loss, train_acc, info='just a test')

    """
    print('Iteration', iteration, end='\t' if info is not None else '\n')
    if info is not None:
        print(info)
    print('Training Loss', train_loss, end='\t' if train_acc is not None else '\n')
    if train_acc is not None:
        print('Training Accuracy', train_acc)
    if valid_loss is not None:
        print('Validation Loss', valid_loss, end='\t' if train_acc is not None else '\n')
    if valid_acc is not None:
        print('Validation Accuracy', valid_acc)
    print()
    if report_file is not None:
        file = open(report_file, 'a')
        file.write('iter' + '\t' + str(iteration) + '\t')
        file.write(str(train_loss) + '\t')
        if train_acc is not None:
            file.write(str(train_acc) + '\t')
        if valid_loss is not None:
            file.write(str(valid_loss) + '\t')
        if valid_acc is not None:
            file.write(str(valid_acc) + '\t')
        file.write('\n')
        file.close()
