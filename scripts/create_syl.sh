#!/bin/tcsh -f

if ($# != 3) then
	echo "arg1 - c_list"
	echo "arg2 - v_list"
	echo "arg3 - lab_file"
	exit(-1)
endif

rm -f cv_list


cat $3 | cut -d " " -f3 | grep -v sil > all_phones

set start = 1
set num_phones = `cat all_phones |wc -l`


#--------To remove the fist phone if it is a vowel
set first_phone = `head -1 all_phones`

set numv = `grep -c -x $first_phone $2`
if ($numv > 0) then
	set start = 2
endif
#------------------

while ($start <= $num_phones)
	set phone = `cat all_phones |head -$start |tail -1`
	set numc = `grep -c  -x $phone $1`
	set numv = `grep -c -x $phone $2`
	set cv = ' '
	if ($numc > 0) then
		set c_flag = 1
		set v_flag = 0
		set consonant = $phone
	else
		set v_flag = 1
		set c_flag = 0
		set vowel = $phone
		set cv = $consonant$vowel
		#echo $consonant $vowel
	endif

	set cv_flag = `echo $cv | wc -w`
	if ($cv_flag > 0) then
		echo ME $cv $consonant $vowel >> led_file1
	endif

	@ start++
end

cp $3 data/temp_syl.lab

HLEd led_file1 data/temp_syl.lab

set n = 1
while ($n < 4) 
	set temp_start = 1
	set temp_num = `cat data/temp_syl.lab | wc -l`

	cat data/temp_syl.lab | cut -d " " -f3 > all_cv

	while ($temp_start <= $temp_num)
        	set temp_phone = `cat all_cv |head -$temp_start |tail -1`
        	set numc = `grep -c  -x $temp_phone $1`
		if ($numc == 0) then
			set temp_syl = $temp_phone
		else 
			echo ME $temp_syl$temp_phone $temp_syl $temp_phone >> led_file2 
		endif
		@ temp_start++
	end

	HLEd led_file2 data/temp_syl.lab

	rm -f led_file1 all_cv all_phones
	@ n++ 
end


