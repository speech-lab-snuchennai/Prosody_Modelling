#!/usr/bin/bash

##############################################################################
#   Purpose: Script for building isolated style subword models from continuous speech
#            utterances that are labeled at the subword level.
#
#   Requirements: wavFolder featureFolder labelFolder subwordList
#
#   
#
#
###############################################################################

currentDir=./
scriptsDir=./scripts

ulimit -s unlimited

if [ $# != 10 ]; then
  echo "Usage: run.sh WaveFileFolder FeatureFolder LabelFolder subWordList hmmDir configFile lexiconDir listDir protoHMMDir resultsDir"
  exit
fi

wavFolder=$1    # Assume wavfiles are in .wav format according to configFile
featureFolder=$2 # Assume features are extracted according to configFile
labFolder=$3     # labFiles give transcription for each file in terms of subword units. Each line gives duration (start, end) and subword unit label.
subwordList=$4
hmmDir=$5
configFile=$6
lexiconDir=$7
listDir=$8
protoHMMDir=$9
resultsDir=$10

rmdir $listDir
mkdir $listDir
rmdir $lexiconDir
mkdir $lexiconDir

ls -1 $wavFolder/*.wav > $listDir/wavlist
ls -1 $featureFolder/*.mfc > $listDir/featureList

paste list/wavlist list/featureList > $listDir/maptable

#We now create the list of subword units using the lab files

cat $labFolder/*.lab | cut -d ' ' -f3 | sort -uniq > $lexiconDir/wordlist

#Now create hmms for each of the subword units.   We need to first give the number of states and mixtures for each subword unit.

$scriptsDir/mix_state_list $lexiconDir/wordlist > mix_state_list

#this creates a file called mix_state_list in the current directory with entries
#
# <subwordUnit> <numMixtures> <numStates>
# Aa              3              5
# Sh              3              5

rmdir $hmmDir
mkdir $hmmDir
mkdir $hmmDir/re-estimated

ls -v $labFolder/*.lab > list/lablist
ls -v $featureFolder/*.mfc > list/trainFeatureList

#Now generate prototype hmms for each of the subword units.

perl $scriptsDir/MakeProtoHMM.pl $protoHMMDir/protoconfig

#Now generate hmm models for each of the subword units in hmmDir

$scriptsDir/modelGeneration.sh mix_state_list $hmmDir

#Now hmms are initialised using HInit and also re-estimated using HRest.

# The re-estimated models are available $hmmDir/re-estimated

# Now copy all the subwordHMMs to lexicon folder

rm $lexiconDir/all_hmm

cat $hmmDir/re-estimated/* >> $lexiconDir/all_hmm

#Now create word_syntax in the $lexiconDir

$scriptsDir/syntax_create $lexiconDir/wordlist

#Now create the network file for the hmms

HParse $lexiconDir/word_syntax $lexiconDir/network

#Now create a dictionary

$scriptsDir/create_dictionary $lexiconDir/wordlist

#This creates a dictionary file called dict in $lexiconDir

#Now we can perform a Viterbi alignment for the training data

HVite -T 1 -C $configFile -S $list/trainFileList -H $lexiconDir/all_hmm -i output -w $lexiconDir/network $lexiconDir/dict $lexiconDir/wordlist

#Now create a master label file for the entire trainFileList

$scriptsDir/mlf $listDir/lablist > $listDir/utts.mlf

#Now compute accuracies and get likelihoods

HResults -I $listDir/utts.mlf $lexiconDir/wordlist $resultsDir/recOutput
