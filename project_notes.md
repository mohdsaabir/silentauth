# SilentAuth – Project Notes
## Main changes made in this iteration

## 1. Db creation
- Db creation is automated it checks if central db present or not during the inital server run if present ignores else creates the db
- Logic for this is writtern in system_controller

## 2. System health check
- Added system health check to ensure weather services are running in all ports or not
- dashboard has overall health check
- verification has health status of its required services
- enrollment has health status of its required services
- For verification and enrollment the health status is rechecked during every 5 seconds

## 3. GUI improvements
- Added a Dashboard which connects to verification,enrollment,system health check,Verification log


## How to run

- As usual cntrl+shift+p in vscode select SilentAuth GUI (ALL) now all enrollment, verification services will be activated except camera service
- Camera service is enabled only during verification and is deactivated after verification

## New Files and its use
- System controller.py : API endpoints to camera service start stop, system health check, db check etc
- System manager.py : Logic of camera service start stop
- verification logger.py : Logic to add the verification status to db
- health check.py : Logic to check the status of each services
- config.py : Made the port as global variable so can be easily used and modified for python files
- static\config.js : Made the port as global variable so can be easily used and modified for html files

  
