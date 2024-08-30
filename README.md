# trust_evaluation

This is a decentralized and manipulation-resistant reputation management approach for distributed networks.

The system is categorized into local and distributed modes. Local mode can be executed by running local->main.py. Distributed mode requires a Hyperledger Fabric environment to be built on the cluster. The cluster environment for this project is contained in the distribution folder.

In order to build the distributed environment, you need to build the basic Hyperledger Fabric first, which can be found at https://hyperledger-fabric.readthedocs.io/en/latest/index.html. In addition, our environment is likewise a blockchain and SDN-based Routing Security project carrier, and in order to ensure that the environment is available, you need to build the OpenFlow environment at the same time. All environment dependencies include Docker, Docker Compose, Hyperledger Fabric, Ryu Controller, mininet , Open vSwitch, and so on. We have provided all the materials required for the basic environment, you need to modify the relevant configuration according to your own environment.

# Description of the files
The 'chain1fabric-docker-multiple.tar.gz package' is the configuration and control file for fabric.

The 'chain1fabric-go-sdk.tar.gz' package is the programming interface file for fabric.

The 'chain1mininet.tar.gz' package is the configuration and control file for mininet.

The 'chain1ryucontroller.tar.gz' package is the configuration and control file for ryu.

# Configure the blockchain network
Initialize the blockchain network and configure each node, see /home/chain*/go/src/github.com/hyperledger/fabric-samples/fabric-docker-multiple-directory.configtx.yaml for details of the configuration file
Deploy CA nodes to generate and distribute digital certificates.
Configure the MSP (Membership Service Provider) to ensure that the nodes can authenticate each other. Configuration files are detailed in the /home/chain*/go/src/github.com/hyperledger/fabric-samples/fabric-docker-multiple directory： crypto-config.yaml

# Configure the SDN controller
Deploy the OpenFlow controller to connect and manage the switches and routers in the network.
Install and configure the controller program to ensure that the SDN controller can handle BGP routing; interact with blockchain clients; and communicate with lower-layer forwarding devices.

# Configure the mininet network
Configure each node's ip address, mac address, and other information. Construct the network topology. See /usr/local/mininet/mininet-file/chain*_out_ovs.py for details.

# Start the service
Start the blockchain network nodes using the docker-compose tool to ensure that the nodes are communicating properly with each other.
Start the SDN controller and load the configured network management modules and BGP modules and modules that go to blockchain interaction.

# Starting and stopping the system
(1) Start the system:

① Start the blockchain network node: `docker-compose -f docker-compose-fabric.yaml up -d`.

② Start the SDN controller: `. /start-sdn-controller.sh`.

③Start the blockchain client.

④Start mininet.

(2) Stop the system:

① Stop SDN controller: `. /stop-sdn-controller.sh`.

② Stop the blockchain network node: `docker-compose -f docker-compose-fabric.yaml down`.
