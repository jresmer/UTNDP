import csv
import os


class LogManager:

    def __init__(self,
                 filename: str="log.csv",
                 add_times_log: bool=False,
                 times_filesname: str="times_log.csv"):

        self.__filename = filename
        self.__fieldnames = ['Instance', 'Population Size', 'Mutation Probability',
                   'Total Fleet Size', 'Min Bus Line/Ind', 'Min Vertices in a Bus Line',
                   'Max Vertices in a Bus Line', 'Transfer Penalty', 'Unreached Stop Penalty',
                   'Best Solution Operator Cost', 'Best Solution Passanger Cost', 'Total Time']
        self.__times_file = add_times_log
        self.__times_filename = times_filesname
        self.__times_fieldnames = ["objFuncs", "nonDominatedSort", "indPreSelection",
                 "hyperplaneUpdateAndNormalization", "associations", "niching", "offspringGeneration"]

    # writes csv file header
    def write_header(self, mode: bool=False) -> bool:

        if not mode:
            filename = self.__filename
            fieldnames = self.__fieldnames
        elif self.__times_file:
            filename = self.__times_filename
            fieldnames = self.__times_fieldnames
        else:
            return False

        # if file does not exist
        if not os.path.exists(filename):
            # create a new file and write
            with open(filename, "wt") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

            return True

        # if file exists and is empty
        elif os.stat(filename).st_size == 0:
            # write header
            with open(filename, "a") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

            return True
        
        return False

    # writes new row
    def write_row(self, 
                  instance: str,
                  population_size: int,
                  mutation_prob: float,
                  fleet_size: int,
                  min_lines: int,
                  min_vertices: int,
                  max_vertices: int,
                  transfer_penalty: int,
                  unreached_stop_penalty: int,
                  best_ind: tuple,
                  start_time: float,
                  end_time: float) -> bool:
        
        # if file exists and is not empty (has at least a header written in it)
        if os.path.exists(self.__filename) and os.stat(self.__filename).st_size != 0:
            # associates contents to columns
            row_content = {
                'Instance': instance,
                'Population Size': population_size,
                'Mutation Probability': mutation_prob,
                'Total Fleet Size': fleet_size,
                'Min Bus Line/Ind': min_lines,
                'Min Vertices in a Bus Line': max_vertices,
                'Max Vertices in a Bus Line': min_vertices,
                'Transfer Penalty': transfer_penalty,
                'Unreached Stop Penalty': unreached_stop_penalty,
                'Best Solution Operator Cost': best_ind[0],
                'Best Solution Passanger Cost': best_ind[1],
                'Total Time': end_time - start_time
            }

            # writes row
            with open(self.__filename, "a", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.__fieldnames)
                writer.writerow(row_content)

            return True

        return False

    def write_times(self, times: dict) -> bool:

        # if file does not exists or is empty
        if not os.path.exists(self.__times_filename) or not os.stat(self.__times_filename).st_size != 0:
            return False
        # if object is not set to have a times log
        if not self.__times_file:
            return False
        
        with open(self.__times_filename, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.__times_fieldnames)
            writer.writerow(times)

        return True
