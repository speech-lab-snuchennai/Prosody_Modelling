#!/bin/tcsh -f


#if($# != 1) then 
#	echo "argument 1 = mix_state_list"
#	exit(-1)
#endif

if($# != 5) then 
	echo "argument 1 = subWordMixStateList hmmDir configFile trainFeatureList labelFileDir"
	exit(-1)
endif


set wordMixList = $1
set hmmDir = $2
set configFile = $3
set trainFeatureList = $4
set labelFileDir = $5

rm -r $hmmDir
rm header
rm footer
set i = 1
set lines = `cat $wordMixList | wc -l`

mkdir $hmmDir

mkdir $hmmDir/re-estimated

while($i <= $lines)
	set word = `cat $wordMixList | head -$i | tail -1 | cut -d " " -f1`
	set mix =  `cat $wordMixList | head -$i | tail -1 | cut -d " " -f2`
	set state = `cat $wordMixList | head -$i | tail -1 | cut -d " " -f3`
        
#	HInit -C config_files/config_feature_file -X lab -S list/train_feature_list -M hmm/ -l $word protos/proto_{$state}s_{$mix}m

	HInit -C $configFile -X lab -S $trainFeatureList -M $hmmDir -L $labelFileDir -l $word protos/proto_{$state}s_{$mix}m


	#mv protos/proto_{$state}s_{$mix}m hmm/proto_{$state}s_{$mix}m 
	
	head -3 $hmmDir/proto_{$state}s_{$mix}m > header
	tail -n +5 $hmmDir/proto_{$state}s_{$mix}m > footer
	cat header > $hmmDir/$word
	echo \~h \"$word\" >> $hmmDir/$word
	cat footer >> $hmmDir/$word
			
#	rename 's/proto_'$state's_'$mix'm/'$word'/' hmm/*


#	replace proto_{$state}s_{$mix}m $word -- hmm/$word         
	
#	HRest -C config_files/feature_config_file -X lab -S list/train_feature_list -M hmm/re-estimated/ -l $word hmm/$word

	HRest -C $configFile -X lab -S $trainFeatureList -M $hmmDir/re-estimated/ -L $labelFileDir -l $word $hmmDir/$word
echo $word model is trained

	@ i++
end
