#include<stdio.h>
#include<stdlib.h>

int main()
{
	printf("Welcome to EarthCooler Project!\n");
	int num = 0, status = 1;
	while(status)
	{
		printf("Type 0 to Launch the CO2Srv, Type 1 to Launch the CO2Client\n");
		status = scanf("%d", &num);
		if(num == 0)
			system("start python_lite.exe co2srv.py");
		else
			system("start python_lite.exe co2client.py");
	}
}
