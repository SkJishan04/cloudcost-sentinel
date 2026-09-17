"""Endpoints for managing and inspecting cloud instances."""

from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.core.exceptions import InstanceNotFoundError
from app.db.models import CloudInstance
from app.schemas.instance import InstanceCostSummary, InstanceCreate, InstanceRead
from app.services import cost_calculator

router = APIRouter(prefix="/api/v1/instances", tags=["instances"])


@router.get("", response_model=list[InstanceRead])
def list_instances(db: Session = Depends(get_db)) -> list[CloudInstance]:
    return db.query(CloudInstance).order_by(CloudInstance.id.asc()).all()


@router.post("", response_model=InstanceRead, status_code=status.HTTP_201_CREATED)
def create_instance(payload: InstanceCreate, db: Session = Depends(get_db)) -> CloudInstance:
    instance = CloudInstance(**payload.model_dump())
    db.add(instance)
    db.commit()
    db.refresh(instance)
    return instance


@router.get("/{instance_id}", response_model=InstanceRead)
def get_instance(instance_id: str, db: Session = Depends(get_db)) -> CloudInstance:
    instance = db.query(CloudInstance).filter(CloudInstance.instance_id == instance_id).first()
    if instance is None:
        raise InstanceNotFoundError(f"Instance '{instance_id}' was not found")
    return instance


@router.get("/{instance_id}/cost", response_model=InstanceCostSummary)
def get_instance_cost(instance_id: str, db: Session = Depends(get_db)) -> InstanceCostSummary:
    instance = db.query(CloudInstance).filter(CloudInstance.instance_id == instance_id).first()
    if instance is None:
        raise InstanceNotFoundError(f"Instance '{instance_id}' was not found")
    return InstanceCostSummary(
        instance_id=instance.instance_id,
        instance_type=instance.instance_type,
        hourly_cost_usd=cost_calculator.get_hourly_cost(instance.instance_type),
        estimated_monthly_cost_usd=cost_calculator.estimate_monthly_cost(instance.instance_type),
        status=instance.status,
    )