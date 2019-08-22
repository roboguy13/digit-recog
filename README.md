Building and running
====================

When this repository is first cloned, you will need to run the following script
to fetch the MNIST files from the MNIST server:

    % ./fetch-images.sh

Next, the following command will install dependences (in a sandbox) and build
the project with -O2:

    % cabal new-build --enable-optimization=2

Finally, this command will run the project executable (note that if you don't
include the optimization flag, it will attempt to rebuild with an optimization
level of -O1 instead of using the existing build with -O2):

    % time cabal new-run --enable-optimization=2

