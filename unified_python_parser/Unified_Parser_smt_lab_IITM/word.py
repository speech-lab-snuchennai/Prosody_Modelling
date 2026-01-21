from indic_unified_parser.uparser import wordparse

output = '/home/sathya-narayanan/speech_new/Perl/wordpronunciation'

# Define the word to be parsed

from word import wordparse


def process_file(input_file_path, output_file_path):
    # Read the input file
    with open(input_file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Split the text into words
    words = text.split()

    # Process each word
    parsed_words = [wordparse(word, 0, 0, 0)[23:] for word in words]


    # Join the parsed words back into a single string
    parsed_text = '\n'.join(parsed_words)

    # Write the parsed text to the output file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(parsed_text)

# Example usage
input_file_path = "/home/sathya-narayanan/speech_new/te.txt"
output_file_path = output
process_file(input_file_path, output_file_path)
