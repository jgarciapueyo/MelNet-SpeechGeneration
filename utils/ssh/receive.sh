# Bash file to receive the logs, model weights and output when perfoming synthesis from the server
# to the local repository. Everything will be saved under the folder redsofa/

# Set environment variables of the server (REMOTEUSER and REMOTEIP) utils/ssh/.env
export $(cat utils/ssh/.env | xargs)

# Receive information from server to local
# check if directory exists and if not, create it
mkdir -p redsofa/
# receive logs
rsync -av -e ssh --progress ${REMOTEUSER}@${REMOTEIP}:/home/${REMOTEUSER}/MelNet-SpeechGeneration/logs redsofa/
# receive checkpoints
rsync -av -e ssh --progress ${REMOTEUSER}@${REMOTEIP}:/home/${REMOTEUSER}/MelNet-SpeechGeneration/models redsofa/
# receive outputs
rsync -av -e ssh --progress ${REMOTEUSER}@${REMOTEIP}:/home/${REMOTEUSER}/MelNet-SpeechGeneration/out redsofa/