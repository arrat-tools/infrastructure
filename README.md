ARRAT AWS CloudFormation Infrastructure deployment

Deploying the EC2 stack

A VPC is required prior to deploying the EC2 instance.  Navigate to the VPC service to verify that you have a VPC set up.  Under Your VPCs, you can click on the existing VPC ID to see it's details, or you can configure a new VPC.  For simplicity, under Actions, you can choose Create default VPC in order to create a VPC with access to the internet.

Once a VPC is selected, choose the desired subnet where the EC2 instance will be deployed.  You will need the VPC ID and the subnet ID in order to deploy the EC2 CloudFormation stack.

Option 1: Deploy using AWS CLI

1: 

Option 2: Deploy using AWS Console

1: Log into AWS and go to the CloudFormation Service

2: Select Upload a template file and click "Choose file" and select the ec2 template.yaml file from your local machine

3: Click Next, give the stack an appropriate name, such as "ARRAT-EC2" and provide the required parameters
    - AMI ID
    - Security Group ID
    - VPC ID
    - Subnet ID
    - Key Pair Name 

4: Click Next and acknowledge and click Create Stack

5: After a few minutes, the stack will be created.  Navigate to the EC2 service, Instances, and note the instance id. This will be needed to deploy the pipeline.

Deploying the Pipeline

After the EC2 stack is deployed and the instance is created, the pipeline step function can be created

Option 1: Deploy using AWS CLI

1: 

Option 2: Deploy using AWS Console

1: Log into AWS and go to the CloudFormation Service

2: Select Upload a template file and click "Choose file" and select the stepfunctions template.yaml file from your local machine

3: Click Next, give the stack an appropriate name, such as "ARRAT-Pipeline" and provide the required parameters
    - Instance ID

4: Click Next and acknowledge and click Create Stack

5: After a few minutes, the stack will be created.  Navigate to the Step Functions service, select the newly created state machine and you can view it's details.  You are now ready to execute the ARRAT pipeline. 
