$ErrorActionPreference = 'Stop'

$sshHost = if ($env:EGODATA_SSH_HOST) { $env:EGODATA_SSH_HOST } else { '' }
$sshUser = if ($env:EGODATA_SSH_USER) { $env:EGODATA_SSH_USER } else { 'Stouch' }
$localPort = if ($env:EGODATA_DB_LOCAL_PORT) { [int]$env:EGODATA_DB_LOCAL_PORT } else { 15432 }
$remoteDbHost = if ($env:EGODATA_DB_REMOTE_HOST) { $env:EGODATA_DB_REMOTE_HOST } else { '' }
$remoteDbPort = if ($env:EGODATA_DB_REMOTE_PORT) { [int]$env:EGODATA_DB_REMOTE_PORT } else { 5432 }

Write-Host "Opening PostgreSQL tunnel 127.0.0.1:$localPort -> $remoteDbHost`:$remoteDbPort via $sshUser@$sshHost"
ssh -o ExitOnForwardFailure=yes -N -L "127.0.0.1:${localPort}:${remoteDbHost}:${remoteDbPort}" "${sshUser}@${sshHost}"
