from indic_unified_parser.uparser import wordparse
import sys
def process_word(word, output_file_path):
    # Process the word
    parsed_word = wordparse(word, 0, 0, 0)
    parsed_word1 = parsed_word[23:]
    
    # Write the parsed word to the output file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(parsed_word1)

# Example usage
word = sys.argv[1] # Replace with your word
output_file_path = '/media/speechlab/Expansion/complete_demo_Wave_modified_Final/Perl/wordpronunciation'
process_word(word, output_file_path)

