#!/bin/bash
# Runs the static analysers and tests

case $1 in
all)
	list_dir=("malpolon/" "malpolon/tests/")
	;;
l)
	list_dir=("malpolon/data" "malpolon/models" "malpolon/plot")
	;;
t)
	list_dir=("malpolon/tests/")
	;;
"")
	list_dir=("malpolon/")
	;;
*)
	list_dir=($1)
	;;
esac

for dir in "${list_dir[@]}";
do
	echo -e "\n\e[1m++++++++++++++++++++++++++++++++++++++"
	echo -e "        Working in \e[92m $dir \e[0m... "
	echo -e "\e[1m++++++++++++++++++++++++++++++++++++++\e[0m"
	liste=$(find $dir -type f -iname '*.py' -not -path '*ipynb_checkpoints/*')
	for fichier in $liste;
	do
		if [[ $fichier != *"__init__.py"* ]] ; then
			if [[ $fichier == *"tests/"* ]]  ; then
				echo -e "Running \e[95m\e[1m pytest\e[0m,\e[95m\e[1m coverage \e[0m on\e[92m $fichier \e[39m..."
				coverage run -m pytest $fichier
				coverage report
			else
				echo -e "Running \e[95m\e[1m Flake8 \e[0m on\e[92m $fichier \e[39m..."
				flake8 $fichier
				echo -e "Running \e[95m\e[1m Pylint \e[0m on\e[92m $fichier \e[39m..."
				pylint $fichier
				echo -e "Running \e[95m\e[1m Pydocstyle \e[0m on\e[92m $fichier \e[39m..."
				pydocstyle $fichier -v
			fi
		fi
	done
done
