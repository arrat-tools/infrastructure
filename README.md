<div align="center"><a name="readme-top"></a></div>

# ARRAT AWS CloudFormation Infrastructure deployment

## Deploying the EC2 stack

> \[!NOTE]
>
> We provide 3 public Amazon Machine Images (AMI) for 3 regions.  These will be used when deploying the EC2 instance.
> 
> - `us-east-1`: ami-08a3b3d19e53d61e3
> - `us-east-2`: ami-00b5f08b5104a8515
> - `us-west-1`: ami-0756ba13d93b566fc 

### Setup the Virtual Private Cloud (VPC)

A VPC is required prior to deploying the EC2 instance.  Navigate to the VPC service to verify that you have a VPC set up.  Under Your VPCs, you can click on the existing VPC ID to see it's details, or you can configure a new VPC.  For simplicity, under Actions, you can choose Create default VPC in order to create a VPC with access to the internet.

**Optional: Create a default VPC**

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/assets/readme/deploy-infra-create-vpc-start.png">
    <img height="240" src="/assets/readme/deploy-infra-create-vpc-start.png" alt="Start to deploy default vpc">
  </picture>
</div>

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/assets/readme/deploy-infra-create-vpc.png">
    <img height="240" src="/assets/readme/deploy-infra-create-vpc.png" alt="Deploy default vpc">
  </picture>
</div>

Once a VPC is selected, choose the desired subnet where the EC2 instance will be deployed.  You will need the VPC ID and the subnet ID in order to deploy the EC2 CloudFormation stack.

*Remember to keep the ID for the VPC (e.g., vpc-) and Subnet (e.g., subnet-) ready since this will be used during deployment*

### Setup the Key Pair and Security Group

In order to connect to your EC2 instance, you will need to set up a key pair and a security group that allows SSH connections from your IP address(es).  

To create a key pair, navigate to the EC2 Console in AWS, and select the Key pairs tab under Network and Security.  Click the Create key pair button, give the key pair an appropriate name, ensure that RSA is the type, and choose the format you need.  Click Create key pair and note the name that you chose.  When prompted, download the key pair file and keep in a safe place, as you will not be able to SSH into the instance without it.

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/assets/readme/deploy-infra-create-key-pair.png">
    <img height="240" src="/assets/readme/deploy-infra-create-key-pair.png" alt="Create key pair">
  </picture>
</div>

To create the security group, navigate to the EC2 Console in AWS, and select the Security Groups tab under Network and Security.  Click Create security group, give it an appropriate name, and select the VPC that you will be using.  Under Inbound rules, click Add rule.  For the new rule, under type, choose SSH, and under Source, choose My IP.  Add a description so you can later identify who is associated with this rule.  Repeat this process for any IP addresses that will need to SSH into the EC2 instance. When finished, click Create security group.  Note the Security group ID of the newly created group (it will being with "sg-") as it will be used to deploy to CloudFormation.

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/assets/readme/deploy-infra-create-security-group.png">
    <img height="240" src="/assets/readme/deploy-infra-create-security-group.png" alt="Create security group">
  </picture>
</div>

*Remember to keep the ID for the Security Group (e.g., sg-) and the name for the Key Pair ready since this will be used during deployment*

### Deploy EC2 instance

Using the IDs from the services set up above, this guide provides the steps for deploying to AWS through the CLI or through the console.

#### Option 1: Deploy using AWS CLI

> \[!NOTE]
>
> If you are unfamiliar with the installation process of the AWS CLI, you can get started with the prerequisites by following the steps [here][back-to-prerequisites]

1: Clone the repository. _Before starting the deployment, clone the [ARRAT Infrastructure repository](https://github.com/arrat-tools/infrastructure) to your local machine._

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/assets/readme/deploy-infra-clone-repo.png">
    <img height="240" src="/assets/readme/deploy-infra-clone-repo.png" alt="Clone ARRAT Infrastructure repository">
  </picture>
</div>

2: From the command line, navigate to the ec2 directory.

```
cd infrastructure/ec2
```

3: Choose the AMI for your desired region and update the parameter overrides with the correct values for your deployment. Run the following command:

```
sam deploy --template-file template.yaml --stack-name arrat-ec2 --parameter-overrides AmiId={choose the ami for your region listed above} SecurityGroupId={your security group id} VPCId={your vpc id} VPCSubnetId={your subnet id} KeyPairName={your key pair name} --capabilities CAPABILITY_NAMED_IAM --profile arrat-cli
```

_If needed, update `arrat-cli` to the profile configured during the prerequisites step_

_You can watch the progress of your deployment from the AWS Cloud Formation web console._

#### Option 2: Deploy using AWS Console

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

*Remember to keep the ID for the EC2 Instance (e.g., i-) by navigating to the Resources tab of your created stack in CloudFormation.*

## Deploying the Pipeline and Step Functions

After the EC2 stack is deployed and the instance is created, the pipeline step function can be created.  Prior to deployment, navigate to the EC2 console in AWS, go to Instances on the left, and note the Instance ID of the intance that was created by the EC2 stack (it will begin with "i-"). 

### Option 1: Deploy using AWS CLI

> \[!NOTE]
>
> If you are unfamiliar with the installation process of the AWS CLI, you can get started with the prerequisites by following the steps [here][back-to-prerequisites]

1: Clone the repository. This step should already be done if you deployed the EC2 instance through the cli.

2: From the command line, navigate to the stepfunctions directory.

```
cd infrastructure/stepfunctions
```

3: update the parameter overrides with the correct values for your deployment, and run the following command:

```
sam deploy --guided --template-file template.yaml --stack-name arrat-stepfunctions --parameter-overrides InstanceId={your instance id generated by the EC2 stack} --capabilities CAPABILITY_NAMED_IAM --profile arrat-cli
```

_If needed, update `arrat-cli` to the profile configured during the prerequisites step_

> \[!NOTE]
> 
> Remember to keep the `SessionInputBucketName`, `SessionOutputBucketName`, and `ARRATStateMachineName` Outputs from the CloudFormation Stack ready since this will be used when deploying the API.
> Also save the `CloudFrontDomain` Output since this will be used when deploying the frontend.

### Option 2: Deploy using AWS Console

1: Log into AWS and go to the CloudFormation Service and navigate to Stacks

2: Select Create stack With new resources, ensure that Choose and existing template is selected, select Upload a template file and click "Choose file" and select the stepfunctions template.yaml file from your local machine

3: Click Next, give the stack an appropriate name, such as "ARRAT-Pipeline" and provide the required parameters
    - Instance ID

4: Click Next, accept the defaults and acknowledgements and click Next

5: Review the settings and click Submit

6: After a few minutes, the stack will be created.  Navigate to the Step Functions service, select the newly created state machine and you can view it's details.  You are now ready to execute the ARRAT pipeline. 

> \[!NOTE]
> 
> Remember to keep the `SessionInputBucketName`, `SessionOutputBucketName`, and `ARRATStateMachineName` Outputs from the CloudFormation Stack ready since this will be used when deploying the API.
> Also save the `CloudFrontDomain` Output since this will be used when deploying the frontend.

### Retrieving Operator Credentials for uploading files to S3

After the CloudFormation templates have been successfully deployed, the vehicle operator can begin uploading the files to S3 that will be processed by the pipeline.  S3 credentials created by the template are securely stored in AWS Secrets Manager.  In order to retrieve these credentials, an Administrator for your AWS account will need to share the Access Key and Secret Access Key that has been stored in Secrets Manager.  

To retreive these keys, the administrator should navigate to AWS Secrets Manager, Secrets, and go to the /arratoperator/credentials/ArratOperator secret.  Select the secret and click Retrieve secret value.  The ACCESS_KEY and SECRET_KEY values should be shared with the vehicle operator.  

[![][back-to-top]](#readme-top)

<!-- Link Groups -->

[back-to-top]: https://img.shields.io/badge/-Back_to_top-151515?style=flat-square
[link-to-repo]: https://github.com/arrat-tools/infrastructure
[back-to-prerequisites]: https://github.com/arrat-tools/deploy/blob/main/guide/00-prerequisites.md
