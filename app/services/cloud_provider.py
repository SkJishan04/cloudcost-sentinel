"""
Cloud provider abstraction.

Real deployments would implement this interface against boto3 (AWS),
azure-mgmt-compute (Azure) or the GCP SDK. For this project a simulated,
DB-backed provider stands in for the real control plane so that the agent's
tool-calling behavior, guardrails and cost math can be exercised end-to-end
and tested deterministically without cloud credentials or billable
side effects. Swapping in a real provider means implementing this same
interface and changing one binding in app/api/deps.py.
"""

from abc import ABC, abstractmethod

from sqlalchemy.orm import Session

from app.core.exceptions import InstanceNotFoundError
from app.db.models import CloudInstance, InstanceStatus
from app.logging_config import get_logger

logger = get_logger(__name__)


class CloudProviderClient(ABC):
    @abstractmethod
    def resize_instance(self, db: Session, instance_id: str, new_instance_type: str) -> CloudInstance: ...

    @abstractmethod
    def shutdown_instance(self, db: Session, instance_id: str) -> CloudInstance: ...

    @abstractmethod
    def start_instance(self, db: Session, instance_id: str) -> CloudInstance: ...

    @abstractmethod
    def get_instance(self, db: Session, instance_id: str) -> CloudInstance: ...


class SimulatedCloudProvider(CloudProviderClient):
    """In-database simulation of an AWS-like control plane."""

    def get_instance(self, db: Session, instance_id: str) -> CloudInstance:
        instance = db.query(CloudInstance).filter(CloudInstance.instance_id == instance_id).first()
        if instance is None:
            raise InstanceNotFoundError(f"Instance '{instance_id}' was not found")
        return instance

    def resize_instance(self, db: Session, instance_id: str, new_instance_type: str) -> CloudInstance:
        instance = self.get_instance(db, instance_id)
        logger.info("Resizing %s: %s -> %s", instance_id, instance.instance_type, new_instance_type)
        instance.instance_type = new_instance_type
        db.commit()
        db.refresh(instance)
        return instance

    def shutdown_instance(self, db: Session, instance_id: str) -> CloudInstance:
        instance = self.get_instance(db, instance_id)
        logger.info("Shutting down %s", instance_id)
        instance.status = InstanceStatus.STOPPED
        db.commit()
        db.refresh(instance)
        return instance

    def start_instance(self, db: Session, instance_id: str) -> CloudInstance:
        instance = self.get_instance(db, instance_id)
        logger.info("Starting %s", instance_id)
        instance.status = InstanceStatus.RUNNING
        db.commit()
        db.refresh(instance)
        return instance


def get_cloud_provider() -> CloudProviderClient:
    """Factory used by services/tools/API layer; single seam for swapping providers."""
    return SimulatedCloudProvider()