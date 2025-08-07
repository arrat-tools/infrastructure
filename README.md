ARRAT AWS CloudFormation Infrastructure deployment

## Deploying the EC2 stack

A VPC is required prior to deploying the EC2 instance.  Navigate to the VPC service to verify that you have a VPC set up.  Under Your VPCs, you can click on the existing VPC ID to see it's details, or you can configure a new VPC.  For simplicity, under Actions, you can choose Create default VPC in order to create a VPC with access to the internet.

Once a VPC is selected, choose the desired subnet where the EC2 instance will be deployed.  You will need the VPC ID and the subnet ID in order to deploy the EC2 CloudFormation stack.

In order to connect to your EC2 instance, you will need to set up a key pair and a security group that allows SSH connections from your IP address(es).  

To create a key pair, navigate to the EC2 Console in AWS, and select the Key pairs tab under Network and Security.  Click the Create key pair button, give the key pair an appropriate name, ensure that RSA is the type, and choose the format you need.  Click Create key pair and note the name that you chose.  When prompted, download the key pair file and keep in a safe place, as you will not be able to SSH into the instance without it.

To create the security group, navigate to the EC2 Console in AWS, and select the Security Groups tab under Network and Security.  Click Create security group, give it an appropriate name, and select the VPC that you will be using.  Under Inbound rules, click Add rule.  For the new rule, under type, choose SSH, and under Source, choose My IP.  Add a description so you can later identify who is associated with this rule.  Repeat this process for any IP addresses that will need to SSH into the EC2 instance. When finished, click Create security group.  Note the Security group ID of the newly created group (it will being with "sg-") as it will be used to deploy to CloudFormation.

Option 1: Deploy using AWS CLI

1: Ensure that the AWS CLI is installed 

2: From the command line, navigate to the ec2 directory.

3: update the parameter overrides with the correct values for your deployment, and run the following command:

aws cloudformation deploy --template-file template.yml --stack-name arrat-ec2 --parameter-overrides AmiId=ami-0481a2c9118bf0a59 SecurityGroupId=sg-0468f4113a4e36d34 VPCId=vpc-00465e9c97c4937fd VPCSubnetId=subnet-07498d8504facc19a KeyPairName=i70_tac_keypair

Option 2: Deploy using AWS Console

1: Log into AWS and go to the CloudFormation Service and navigate to Stacks

2: Select Create stack With new resources, ensure that Choose and existing template is selected, select Upload a template file and click "Choose file" and select the ec2 template.yaml file from your local machine

3: Click Next, give the stack an appropriate name, such as "arrat-ec2" and provide the required parameters
    - AMI ID
    - Security Group ID
    - VPC ID
    - Subnet ID
    - Key Pair Name 

4: Click Next, accept the defaults and acknowledgements and click Next

5: Review the settings and click Submit

6: After a few minutes, the stack will be created.  Navigate to the EC2 service, Instances, and note the instance id. This will be needed to deploy the pipeline.

## Deploying the Pipeline

After the EC2 stack is deployed and the instance is created, the pipeline step function can be created.  Prior to deployment, navigate to the EC2 console in AWS, go to Instances on the left, and note the Instance ID of the intance that was created by the EC2 stack (it will begin with "i-"). 

Option 1: Deploy using AWS CLI

1: Ensure that the AWS CLI is installed 

2: From the command line, navigate to the stepfunctions directory.

3: update the parameter overrides with the correct values for your deployment, and run the following command:

aws cloudformation deploy --template-file template.yml --stack-name arrat-ec2 --parameter-overrides InstanceId=i-0b0b76f95e728f27f 

Option 2: Deploy using AWS Console

1: Log into AWS and go to the CloudFormation Service and navigate to Stacks

2: Select Create stack With new resources, ensure that Choose and existing template is selected, select Upload a template file and click "Choose file" and select the stepfunctions template.yaml file from your local machine

3: Click Next, give the stack an appropriate name, such as "ARRAT-Pipeline" and provide the required parameters
    - Instance ID

4: Click Next, accept the defaults and acknowledgements and click Next

5: Review the settings and click Submit

6: After a few minutes, the stack will be created.  Navigate to the Step Functions service, select the newly created state machine and you can view it's details.  You are now ready to execute the ARRAT pipeline. 
