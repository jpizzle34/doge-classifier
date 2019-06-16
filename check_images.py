#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# TODO: 0. Fill in your information in the programming header below
# PROGRAMMER: John Zambrano
# DATE CREATED: 8th June 2019
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir  <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from datetime import datetime
from time import sleep
from os import listdir
import itertools as it

# Imports classifier function for using CNN to classify images
from classifier import classifier

# Imports print functions that check the lab
from print_functions_for_lab_checks import check_calculating_results
from print_functions_for_lab_checks import check_classifying_images
from print_functions_for_lab_checks import check_classifying_labels_as_dogs
from print_functions_for_lab_checks import check_command_line_arguments
from print_functions_for_lab_checks import check_creating_pet_image_labels

# Main program function defined below


def main():

    start_time = datetime.now()

    in_arg = get_input_args()
    # check_command_line_arguments(in_arg)

    answers_dic = get_pet_labels(in_arg.dir)
    #check_creating_pet_image_labels(answers_dic)

    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)
    #check_classifying_images(result_dic)

    adjust_results4_isadog(result_dic, "dognames.txt")
    #check_classifying_labels_as_dogs(result_dic)

    results_stats_dic = calculates_results_stats(result_dic)
    #check_calculating_results(result_dic, results_stats_dic)

    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    end_time = datetime.now()
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", tot_time)


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
                      pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                            'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line
     arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    parser = argparse.ArgumentParser()
    # Arg1: dir - path to pet_images folder
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='path to images folder, default: "pet_images/"')
    # Arg2: arch - CNN model architecture type - vgg, alexnet, resnet
    parser.add_argument('--arch', type=str, default='vgg',
                        help='chosen model: vgg (default), resnet, alexnet')
    # Arg3: dogfile - Text file that contains all labels associated to dogs
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='txt file with dog names e.g. "dognames.txt"')
    return parser.parse_args()


def get_pet_words(iterable):
    """Generator helper function to extract pet words from a file name."""
    for elem in iterable:
        words = ""
        for word in elem:
            if word.isalpha():
                words += word.lower() + ' '
        yield words.strip()


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these labels as petlabel_dic. This is used to check
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                             classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                                     Labels (as value)
    """
    # read in filenames in pet directory
    dir_list = listdir(image_dir)
    # Define empty dict for pet_labels
    pet_labels_dict = []
    # generator expression to filter out non-images and split words in filename
    petimages = (filename.split("_") for filename in dir_list
                 if filename[0].isalpha())
    # generator expression to build pet_labels
    pet_labels = (pet for pet in get_pet_words(petimages))
    pet_labels_dict = dict(it.zip_longest(dir_list, pet_labels))
    # TODO: Do we need to check for duplicate filenames?

    return pet_labels_dict


def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
    Uses classifier function in classifier.py
      images_dir - The (full) path to the folder of images that are to be
                               classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                    that classify what's in the image, where its key is the
                    pet image filename & its value is pet image label where
                    label is lowercase with space between each word in label
      model - pretrained CNN whose architecture is indicated by this parameter,
                      values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
                     (index)idx 0 = pet image label (string)
                            idx 1 = classifier label
                            idx 2 = 1/0 (int)
                    where 1 = match between pet image and classifer labels and
                          0 = no match between labels
    """
    results_dic = dict()
    # TODO: Replace with generators? or alternate solution. Or both?
    for key in petlabel_dic:

        model_label = classifier(images_dir+key, model)
        model_label = model_label.strip().lower()
        # print(model_label)
        truth = petlabel_dic[key]
        found = model_label.find(truth)
        # if ground truth label is anywhere in model_label
        if found >= 0:
            # CASE1: truth and model label are single terms that match.
            if ((found == 0 and len(truth) == len(model_label)) or
                # CASE2: truth matches a whole term in model label AND
                (((found == 0) or (model_label[found - 1] == " ")) and
                    # the term is at the end of the model label OR
                    ((found + len(truth) == len(model_label)) or
                     # is directly before a comma or blank space
                     (model_label[found + len(truth): found+len(truth) + 1]
                      in (",", " "))
                     )
                 )
                ):
                # Is a match. Stand-alone term.
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 1]
            # Not a match. found within a term. Not on its own.
            else:
                if key not in results_dic:
                    results_dic[key] = [truth, model_label, 0]
        # Not a match. No occurence of term at all.
        else:
            if key not in results_dic:
                results_dic[key] = [truth, model_label, 0]

    return results_dic


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
                    (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                    0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                    'as-a' dog and 0 = Classifier classifies image
                    'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line.
                Dog names are all in lowercase with spaces separating the
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
               None - results_dic is mutable data type so no return needed.
    """
    dognames_dic = dict()

    # read dogsfile into dognames_dic
    with open(dogsfile, "r") as f:
        line = f.readline()
        while line != "":
            line = line.rstrip()
            # Check for duplicate entries
            if line not in dognames_dic:
                dognames_dic[line] = 1
            else:
                print("**Warning: Duplicate dognames:", line)

            line = f.readline()

    # Extends results_dic with ix3 and ix4
    for key in results_dic:
        # Pet image label IS a dog
        if results_dic[key][0] in dognames_dic:
            # Classifier label IS a dog
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((1, 1))
            else:
                results_dic[key].extend((1, 0))
        # Pet image label IS NOT a dog
        else:
            # Classifier label IS a dog or NOT a dog
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((0, 1))
            else:
                results_dic[key].extend((0, 0))


# TODO: Is there a better way to validate our results? Precision/recall

def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
                    (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                                    0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                    'as-a' dog and 0 = Classifier classifies image
                    'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                    percentage or a count) where the key is the statistic's
                    name (starting with 'pct' for percentage or 'n' for count)
                    and the value is the statistic's value
    """
    results_stats = dict()
    # COUNTS STATS
    # Number of images - TOTAL SIZE OF TEST-SET
    results_stats['n_images'] = len(results_dic)
    # TOTAL NUMBER OF TRUE DOGS IN TEST-SET
    results_stats['n_dogs_img'] = sum(
        1 for value in results_dic.values() if value[3] == 1)
    # Number of image labels -> classifier label matches
    results_stats['n_match'] = sum(
        1 for value in results_dic.values() if value[2] == 1)
    # TRUE POSITIVE: Number of both image and clf labels are dogs
    results_stats['n_correct_dogs'] = sum(
        1 for value in results_dic.values() if (value[3] & value[4]) == 1)
    # TRUE NEGATIVE: Number of correct NOT DOG classifications
    results_stats['n_correct_notdogs'] = sum(
        1 for value in results_dic.values()
        if(value[3] == 0 and value[4] == 0))
    # Number of correct breed.
    results_stats['n_correct_breed'] = sum(
        1 for value in results_dic.values() if (value[2] & value[3]) == 1)

    # CALCULATED COUNTS
    # TOTAL NUMBER OF ACTUAL NON-DOGS
    results_stats['n_notdogs_img'] = results_stats['n_images'] - \
        results_stats['n_dogs_img']
    # FALSE POSITIVES: Classifier outputs dog when not-a-dog
    results_stats['n_incorrect_dogs'] = results_stats['n_notdogs_img'] - \
        results_stats['n_correct_notdogs']
    # FALSE NEGATIVES: Classifier outputs not a dog when A-DOG
    results_stats['n_incorrect_notdogs'] = results_stats['n_dogs_img'] - \
        results_stats['n_correct_dogs']

    # PERCENTAGE STATS
    # Correctly classified PETS
    results_stats['pct_match'] = results_stats['n_match'] / \
        results_stats['n_images'] * 100.0
    # PRECISION: out of all dogs how many were correctly classified as dogs
    results_stats['pct_correct_dogs'] = results_stats['n_correct_dogs'] / \
        results_stats['n_dogs_img'] * 100.0

    # Precision: ratio of correct not-dog classifications
    if results_stats['n_notdogs_img'] > 0:
        results_stats['pct_correct_notdogs'] = \
            results_stats['n_correct_notdogs'] / \
            results_stats['n_notdogs_img'] * 100.0
    else:
        results_stats['pct_correct_notdogs'] = 0.0

    # Correct Breed
    results_stats['pct_correct_breed'] = results_stats['n_correct_breed'] / \
        results_stats['n_dogs_img'] * 100.0

    # PRECISION: ratio of true positives to  total predicted positive
    results_stats['precision'] = results_stats['n_correct_dogs'] / \
        (results_stats['n_correct_dogs'] +
            results_stats['n_incorrect_dogs']) * 1.0
    # RECALL: ratio of true positives to total number of actual dogs
    results_stats['recall'] = results_stats['n_correct_dogs'] / \
        (results_stats['n_correct_dogs'] +
         results_stats['n_incorrect_notdogs'])

    return results_stats


def print_results(results_dic, results_stats, model,
                  print_incorrect_dogs=False,
                  print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
                    (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                        0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                    'as-a' dog and 0 = Classifier classifies image
                    'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                                        percentage or a count) where the key
                                        is the statistic's
                                        name (starting with 'pct' for
                                        percentage or 'n' for count)
                                        and the value is the statistic's value
      model - pretrained CNN whose architecture is indicated by this parameter,
                      values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                            False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                            False doesn't print anything(default) (bool)
    Returns:
               None - simply printing results.
    """
    # Prints summary statistics over the run
    print("\n\n*** Results Summary for CNN Model Architecture", model.upper(),
          "***")
    print("%20s: %3d" % ('N Images', results_stats['n_images']))
    print("%20s: %3d" % ('N Dog Images', results_stats['n_dogs_img']))
    print("%20s: %3d" % ('N Not-Dog Images', results_stats['n_notdogs_img']))

    # Prints summary statistics (percentages) on Model Run
    print(" ")
    for key in results_stats:
        if key[0:3] == "pct":
            print("%20s: %5.1f" % (key, results_stats[key]))
    print("\n**BINARY CLASSIFICATION METRICS: 'Dog/Not-a-Dog'**")
    print("%20s: %5.4f" % ("PRECISION", results_stats['precision']))
    print("%20s: %5.4f" % ("RECALL", results_stats['recall']))

    # IF print_incorrect_dogs == True AND there were images incorrectly
    # classified as dogs or vice versa - print out these cases
    if (print_incorrect_dogs and
            ((results_stats['n_correct_dogs'] +
              results_stats['n_correct_notdogs'])
             != results_stats['n_images'])):

        print("\nINCORRECT Dog/NOT Dog Assignments:")

        # process through results dict, printing incorrectly classified dogs
        for key in results_dic:

            # Pet Image Label is a Dog - Classified as NOT-A-DOG -OR-
            # Pet Image Label is NOT-a-Dog - Classified as a-DOG
            if sum(results_dic[key][3:]) == 1:
                print("Real: %-26s   Classifier: %-30s" % (results_dic[key][0],
                                                           results_dic[key][1])
                      )

    # IF print_incorrect_breed == True AND there were dogs whose breeds
    # were incorrectly classified - print out these cases
    if (print_incorrect_breed and
            (results_stats['n_correct_dogs'] !=
             results_stats['n_correct_breed'])):

        print("\nINCORRECT Dog Breed Assignment:")

        # process through results dict, printing incorrectly classified breeds
        for key in results_dic:

            # Pet Image Label is-a-Dog, classified as-a-dog but is WRONG breed
            if (sum(results_dic[key][3:]) == 2 and
                    results_dic[key][2] == 0):
                print("Real: %-26s   Classifier: %-30s" % (results_dic[key][0],
                                                           results_dic[key][1])
                      )


# Call to main function to run the program
if __name__ == "__main__":
    main()
