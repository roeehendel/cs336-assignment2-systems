# Sync results from remote to local ./tmp directory
echo "Syncing results from remote to local ./tmp directory..."
rsync -avz -e "ssh -p 4040" ${REMOTE_HOST}:~/cs336-assignment2-systems/tmp/ ./tmp/