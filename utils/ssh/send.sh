# Bash file to send the current project to the server to perform training or synthesis

# Set environment variables of the server (REMOTEUSER and REMOTEIP) stored in utils/ssh/.env
export $(cat utils/ssh/.env | xargs)

# Send information from local to server except directories declared in utils/ssh/.sendignore
rsync -av -e ssh --progress $(cat utils/ssh/.sendignore | xargs) . ${REMOTEUSER}@${REMOTEIP}:/home/${REMOTEUSER}/MelNet-SpeechGeneration

