@echo off
set /a count=0
set /a max_runs=50

:loop
call npm start
if errorlevel 1 (
    echo The command failed with errorlevel %errorlevel% on iteration %count%.
    goto continue
)
goto :eof

:retry
echo sleep 5 sec
timeout /t 5 >nul
goto loop

echo Command executed successfully on iteration %count%.

:continue
set /a count+=1
if %count% lss %max_runs% goto retry

echo Finished running the command.

goto :eof