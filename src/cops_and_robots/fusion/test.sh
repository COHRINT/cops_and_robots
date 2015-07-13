ssh odroid@deckard 'host `echo ${SSH_CLIENT%% *}`'
echo $TEST
